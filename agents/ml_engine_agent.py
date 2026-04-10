"""
JR MineralForge – ML Engine Agent
====================================
Implements the full ML prospectivity pipeline with:
  - Anti-noise: Isolation Forest outlier removal, data validation
  - Anti-overfitting: Spatial CV (block), ensemble models with regularisation,
    SHAP feature selection, early stopping, uncertainty quantification (bootstrap)
  - Models: Balanced Random Forest, XGBoost, LightGBM
  - Calibration: Platt/isotonic
  - MLflow experiment tracking
"""

from __future__ import annotations

import warnings
import json
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("SHAP not installed — feature selection step will be skipped")

from config.settings import ML_CONFIG, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MODELS_DIR
from utils.logging_utils import get_logger

log = get_logger(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ─────────────────────────────────────────────────────────────────
# Spatial Cross-Validation
# ─────────────────────────────────────────────────────────────────

class SpatialBlockCV:
    """
    Block spatial cross-validation that partitions the study area into
    geographic blocks and assigns each sample to a fold by its block.
    Prevents spatial data leakage between train and test sets.
    """

    def __init__(
        self,
        n_folds: int = ML_CONFIG.n_spatial_folds,
        buffer_km: float = ML_CONFIG.spatial_buffer_km,
        random_state: int = ML_CONFIG.random_state,
    ):
        self.n_folds = n_folds
        self.buffer_km = buffer_km
        self.random_state = random_state

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        coords: Optional[pd.DataFrame] = None,
    ):
        """
        Yield (train_idx, test_idx) tuples.
        coords: DataFrame with 'x' and 'y' columns in projected coordinates (metres).
        Falls back to k-fold if no coordinates provided.
        """
        if coords is None or "x" not in coords.columns or "y" not in coords.columns:
            log.warning("No spatial coordinates — falling back to stratified k-fold")
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            for train, test in skf.split(X, y):
                yield train, test
            return

        # Build spatial blocks by dividing bounding box
        xs, ys = coords["x"].values, coords["y"].values
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        n_blocks_x = int(np.ceil(np.sqrt(self.n_folds)))
        n_blocks_y = int(np.ceil(self.n_folds / n_blocks_x))

        x_edges = np.linspace(x_min, x_max, n_blocks_x + 1)
        y_edges = np.linspace(y_min, y_max, n_blocks_y + 1)

        block_ids = np.zeros(len(X), dtype=int)
        bid = 0
        for i in range(n_blocks_x):
            for j in range(n_blocks_y):
                mask = (
                    (xs >= x_edges[i]) & (xs < x_edges[i + 1]) &
                    (ys >= y_edges[j]) & (ys < y_edges[j + 1])
                )
                block_ids[mask] = bid % self.n_folds
                bid += 1

        for fold in range(self.n_folds):
            test_idx = np.where(block_ids == fold)[0]
            if self.buffer_km > 0:
                # Apply exclusion buffer: remove train samples within buffer_km of test
                buffer_m = self.buffer_km * 1000
                candidate_train = np.where(block_ids != fold)[0]
                test_xs, test_ys = xs[test_idx], ys[test_idx]
                train_xs, train_ys = xs[candidate_train], ys[candidate_train]
                # distance from each train point to nearest test point
                min_dists = np.array([
                    np.min(np.sqrt((tx - test_xs) ** 2 + (ty - test_ys) ** 2))
                    for tx, ty in zip(train_xs, train_ys)
                ])
                train_idx = candidate_train[min_dists > buffer_m]
            else:
                train_idx = np.where(block_ids != fold)[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx


# ─────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────

def engineer_geophysical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived geophysical features from raw grid values.
    Expects columns: tmi, gravity, k_pct, eth, eu
    Generates IOCG-relevant proxy features.
    """
    df = df.copy()

    # Magnetic derivatives (first-order approximation from gridded values)
    if "tmi" in df.columns:
        df["tmi_abs"] = np.abs(df["tmi"])
        df["tmi_z_score"] = (df["tmi"] - df["tmi"].mean()) / (df["tmi"].std() + 1e-8)

    # Gravity features
    if "gravity" in df.columns:
        df["gravity_z_score"] = (df["gravity"] - df["gravity"].mean()) / (df["gravity"].std() + 1e-8)
        if "tmi" in df.columns:
            df["mag_grav_ratio"] = df["tmi"] / (df["gravity"].abs() + 1e-8)

    # Radiometric IOCG proxies
    if "k_pct" in df.columns and "eth" in df.columns:
        df["k_eth_ratio"] = df["k_pct"] / (df["eth"] + 1e-8)  # potassic alteration proxy
        df["f_parameter"] = 2 * df["k_pct"] + df["eth"] / 4
        if "eu" in df.columns:
            df["f_parameter"] += df["eu"] / 2

    # Geochemistry proxies (if present)
    for col in ["cu_ppm", "au_ppb", "fe_pct"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    return df


def select_features_by_shap(
    model,
    X: pd.DataFrame,
    fraction: float = ML_CONFIG.shap_feature_fraction,
) -> List[str]:
    """Use SHAP to identify top-fraction features. Returns list of selected column names."""
    if not HAS_SHAP:
        return list(X.columns)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # binary classification
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        # Rank features
        ranking = pd.Series(mean_abs_shap, index=X.columns).sort_values(ascending=False)
        n_keep = max(1, int(len(ranking) * fraction))
        selected = list(ranking.index[:n_keep])
        log.info(f"SHAP selected {len(selected)}/{len(X.columns)} features")
        return selected
    except Exception as e:
        log.warning(f"SHAP feature selection failed: {e}")
        return list(X.columns)


# ─────────────────────────────────────────────────────────────────
# Outlier Detection (Anti-Noise)
# ─────────────────────────────────────────────────────────────────

def remove_outliers_isolation_forest(
    X: pd.DataFrame,
    contamination: float = ML_CONFIG.isolation_forest_contamination,
    random_state: int = ML_CONFIG.random_state,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Apply Isolation Forest to detect and remove outlier samples.
    Returns (cleaned_X, inlier_mask).
    Anti-noise measure: removes anomalous, potentially erroneous data points.
    """
    log.info(f"Isolation Forest outlier detection (contamination={contamination})")
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    preds = iso.fit_predict(X.fillna(X.median()))
    inlier_mask = preds == 1
    n_removed = (~inlier_mask).sum()
    log.info(f"  Removed {n_removed} outlier samples ({n_removed/len(X)*100:.1f}%)")
    return X[inlier_mask], inlier_mask


# ─────────────────────────────────────────────────────────────────
# Main ML Engine
# ─────────────────────────────────────────────────────────────────

class ProspectivityMLEngine:
    """
    End-to-end ML engine for mineral prospectivity mapping.
    Implements the full anti-noise + anti-overfitting pipeline.
    """

    def __init__(self, config=ML_CONFIG):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.feature_cols: List[str] = []
        self.cv_results: Dict[str, Any] = {}
        self.shap_feature_importance: Optional[pd.Series] = None

    # ── Data Validation ───────────────────────────────────────────

    def validate_data(self, df: pd.DataFrame, target_col: str = "label") -> pd.DataFrame:
        """Basic data validation and cleaning."""
        log.info("Validating training data …")
        # Remove rows with all-NaN features
        feat_cols = [c for c in df.columns if c != target_col]
        df = df.dropna(subset=feat_cols, how="all")
        # Fill remaining NaN with median (feature-wise)
        df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())
        # Clip extreme values (3.5-sigma)
        for col in feat_cols:
            mu, sigma = df[col].mean(), df[col].std()
            if sigma > 0:
                df[col] = df[col].clip(mu - 3.5 * sigma, mu + 3.5 * sigma)
        # Check class balance
        if target_col in df.columns:
            counts = df[target_col].value_counts()
            log.info(f"  Class distribution: {counts.to_dict()}")
        log.info(f"  After validation: {len(df)} samples, {len(feat_cols)} features")
        return df

    # ── Training ──────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "label",
        coord_df: Optional[pd.DataFrame] = None,
    ) -> "ProspectivityMLEngine":
        """
        Full training pipeline:
        1. Validate data
        2. Engineer features
        3. Remove outliers (Isolation Forest)
        4. Spatial CV to estimate performance
        5. Train ensemble (RF + XGBoost + LGB) on full data
        6. SHAP feature selection
        7. Calibrate probabilities
        8. Bootstrap uncertainty estimation
        9. Log to MLflow
        """
        with mlflow.start_run(experiment_id=self._get_or_create_experiment()):
            # Step 1: Validate
            df = self.validate_data(df, target_col)

            # Step 2: Feature engineering
            df = engineer_geophysical_features(df)

            # Step 3: Outlier removal (apply to features only, not label)
            feat_cols = [c for c in df.columns if c != target_col]
            X = df[feat_cols]
            y = df[target_col]
            X_clean, inlier_mask = remove_outliers_isolation_forest(X)
            y_clean = y[inlier_mask]
            if coord_df is not None:
                coord_df = coord_df[inlier_mask]

            # Step 4: Spatial CV
            self.feature_cols = list(X_clean.columns)
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_clean),
                columns=self.feature_cols
            )
            cv_aucs = self._spatial_cv(X_scaled, y_clean, coord_df)
            self.cv_results = {
                "mean_auc": float(np.mean(cv_aucs)),
                "std_auc": float(np.std(cv_aucs)),
                "fold_aucs": [float(v) for v in cv_aucs],
            }
            log.info(f"Spatial CV AUC: {self.cv_results['mean_auc']:.4f} ± {self.cv_results['std_auc']:.4f}")
            mlflow.log_metrics({"cv_mean_auc": self.cv_results["mean_auc"],
                                "cv_std_auc": self.cv_results["std_auc"]})

            # Step 5: Train ensemble on full data
            X_np = X_scaled.values
            y_np = y_clean.values
            self._train_ensemble(X_np, y_np, X_scaled)

            # Step 6: SHAP feature selection (uses RF model)
            if HAS_SHAP and "rf" in self.models:
                selected_feats = select_features_by_shap(
                    self.models["rf_base"], X_scaled, self.config.shap_feature_fraction
                )
                # Retrain on selected features only
                X_selected = X_scaled[selected_feats]
                self._train_ensemble(X_selected.values, y_np, X_selected)
                self.feature_cols = selected_feats

            # Step 7: Bootstrap uncertainty
            self.bootstrap_models = self._bootstrap_train(X_scaled.values, y_np)

            # Save models
            self._save_models()
            mlflow.log_param("n_features", len(self.feature_cols))
            mlflow.log_param("n_train_samples", len(y_np))
            mlflow.log_param("positive_rate", float(y_np.mean()))

        return self

    def _get_or_create_experiment(self) -> str:
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            return mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
        return experiment.experiment_id

    def _build_rf(self):
        return RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            class_weight="balanced",
            random_state=self.config.random_state,
            n_jobs=-1,
        )

    def _build_xgb(self):
        return xgb.XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            reg_alpha=self.config.xgb_reg_alpha,
            reg_lambda=self.config.xgb_reg_lambda,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            scale_pos_weight=10,
            random_state=self.config.random_state,
            eval_metric="aucpr",
            use_label_encoder=False,
            verbosity=0,
        )

    def _build_lgb(self):
        return lgb.LGBMClassifier(
            n_estimators=self.config.lgb_n_estimators,
            num_leaves=self.config.lgb_num_leaves,
            learning_rate=self.config.lgb_learning_rate,
            reg_alpha=self.config.lgb_reg_alpha,
            reg_lambda=self.config.lgb_reg_lambda,
            min_child_samples=self.config.lgb_min_child_samples,
            class_weight="balanced",
            random_state=self.config.random_state,
            n_jobs=-1,
            verbose=-1,
        )

    def _train_ensemble(self, X: np.ndarray, y: np.ndarray, X_df: pd.DataFrame):
        """Train RF + XGBoost + LGB and build calibrated ensemble."""
        log.info("Training Balanced Random Forest …")
        rf_base = self._build_rf()
        rf_base.fit(X, y)
        rf_cal = CalibratedClassifierCV(rf_base, method=self.config.calibration_method, cv=3)
        rf_cal.fit(X, y)
        self.models["rf_base"] = rf_base
        self.models["rf"] = rf_cal

        log.info("Training XGBoost …")
        xgb_model = self._build_xgb()
        xgb_model.fit(X, y)
        xgb_cal = CalibratedClassifierCV(xgb_model, method=self.config.calibration_method, cv=3)
        xgb_cal.fit(X, y)
        self.models["xgb"] = xgb_cal

        log.info("Training LightGBM …")
        lgb_model = self._build_lgb()
        lgb_model.fit(X, y)
        lgb_cal = CalibratedClassifierCV(lgb_model, method=self.config.calibration_method, cv=3)
        lgb_cal.fit(X, y)
        self.models["lgb"] = lgb_cal

    def _spatial_cv(
        self, X: pd.DataFrame, y: pd.Series, coords: Optional[pd.DataFrame]
    ) -> List[float]:
        """Run spatial block CV and return fold AUC scores."""
        cv = SpatialBlockCV()
        aucs = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, coords)):
            X_tr, X_te = X.iloc[train_idx].values, X.iloc[test_idx].values
            y_tr, y_te = y.iloc[train_idx].values, y.iloc[test_idx].values
            if len(np.unique(y_te)) < 2:
                continue
            m = self._build_rf()
            m.fit(X_tr, y_tr)
            proba = m.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, proba)
            aucs.append(auc)
            log.info(f"  Fold {fold + 1}: AUC = {auc:.4f}")
        return aucs

    def _bootstrap_train(self, X: np.ndarray, y: np.ndarray, n: int = ML_CONFIG.n_bootstrap) -> List:
        """Train N bootstrap RF models for uncertainty quantification."""
        log.info(f"Training {n} bootstrap models for uncertainty …")
        bootstrap_models = []
        n_samples = len(y)
        rng = np.random.default_rng(self.config.random_state)
        for i in range(n):
            idx = rng.choice(n_samples, n_samples, replace=True)
            m = RandomForestClassifier(
                n_estimators=50, max_depth=8, min_samples_leaf=5,
                class_weight="balanced", random_state=int(rng.integers(1e6)), n_jobs=1,
            )
            m.fit(X[idx], y[idx])
            bootstrap_models.append(m)
        return bootstrap_models

    # ── Prediction ────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Ensemble probability prediction (mean of calibrated RF + XGB + LGB).
        Applies scaler and feature selection consistently.
        """
        X_eng = engineer_geophysical_features(X)
        X_feat = X_eng[self.feature_cols].fillna(X_eng[self.feature_cols].median())
        X_scaled = self.scaler.transform(X_feat)
        proba_rf = self.models["rf"].predict_proba(X_scaled)[:, 1]
        proba_xgb = self.models["xgb"].predict_proba(X_scaled)[:, 1]
        proba_lgb = self.models["lgb"].predict_proba(X_scaled)[:, 1]
        return (proba_rf + proba_xgb + proba_lgb) / 3.0

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (mean_proba, std_uncertainty) from bootstrap ensemble.
        std_uncertainty quantifies epistemic uncertainty per prediction.
        """
        X_eng = engineer_geophysical_features(X)
        X_feat = X_eng[self.feature_cols].fillna(X_eng[self.feature_cols].median())
        X_scaled = self.scaler.transform(X_feat)
        bootstrap_probas = np.array([
            m.predict_proba(X_scaled)[:, 1] for m in self.bootstrap_models
        ])
        return bootstrap_probas.mean(axis=0), bootstrap_probas.std(axis=0)

    def compute_shap_values(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Compute SHAP values for interpretability."""
        if not HAS_SHAP or "rf_base" not in self.models:
            return None
        X_eng = engineer_geophysical_features(X)
        X_feat = X_eng[self.feature_cols].fillna(X_eng[self.feature_cols].median())
        X_scaled = pd.DataFrame(self.scaler.transform(X_feat), columns=self.feature_cols)
        explainer = shap.TreeExplainer(self.models["rf_base"])
        return explainer.shap_values(X_scaled)

    # ── Persistence ───────────────────────────────────────────────

    def _save_models(self):
        MODELS_DIR.mkdir(exist_ok=True)
        with open(MODELS_DIR / "ml_engine.pkl", "wb") as f:
            pickle.dump({
                "models": self.models,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "cv_results": self.cv_results,
            }, f)
        log.info(f"Models saved → {MODELS_DIR / 'ml_engine.pkl'}")

    @classmethod
    def load(cls) -> "ProspectivityMLEngine":
        engine = cls()
        pkl_path = MODELS_DIR / "ml_engine.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"No saved model at {pkl_path}; run fit() first")
        with open(pkl_path, "rb") as f:
            state = pickle.load(f)
        engine.models = state["models"]
        engine.scaler = state["scaler"]
        engine.feature_cols = state["feature_cols"]
        engine.cv_results = state["cv_results"]
        log.info("ML engine loaded from disk")
        return engine
