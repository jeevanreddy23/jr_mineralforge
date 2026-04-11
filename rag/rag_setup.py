"""
JR MineralForge – RAG Setup
==============================
Initialises and manages the FAISS / Chroma vector store and the RAG chain.
Handles document loading, chunking, embedding, and retrieval with score threshold
(anti-noise: discard low-relevance chunks).

Knowledge base includes:
  - OZ Minerals Explorer Challenge winner analyses
  - Geoscience Australia / SARIG technical reports (PDFs, text)
  - Geological literature on IOCG mineral systems
  - User-provided documents
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, CSVLoader, DirectoryLoader
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import (
    EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_BATCH_SIZE,
    VECTOR_STORE_BACKEND, VECTOR_STORE_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_K, RETRIEVAL_SCORE_THRESHOLD,
    OZ_MINERALS_CHALLENGE_WINNERS, WINNERS_KB_PATH,
    BRAND_NAME, TEAM_NAME, BRAND_HEADER, WINNERS_CONTEXT,
    DATA_DIR,
)
from utils.logging_utils import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embeddings instance."""
    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"batch_size": EMBEDDING_BATCH_SIZE, "normalize_embeddings": True},
    )


# ─────────────────────────────────────────────────────────────────
# Winners Synthetic Knowledge Documents
# ─────────────────────────────────────────────────────────────────

def _build_winner_documents() -> List[Document]:
    """
    Generate structured Documents from the hard-coded OZ Minerals winner database.
    Each winner becomes a rich Document with metadata for retrieval.
    """
    docs = []
    for w in OZ_MINERALS_CHALLENGE_WINNERS:
        tactics_text = "\n".join(f"  - {t}" for t in w["key_tactics"])
        content = (
            f"Team: {w['team']}\n"
            f"Prize: {w['prize']} | Year: {w['year']} | Challenge: {w['challenge']}\n\n"
            f"Key Winning Tactics:\n{tactics_text}\n\n"
            f"How JR MineralForge improves upon {w['team']}:\n"
            f"Team JR integrates SARIG and Geoscience Australia open data with robust "
            f"spatial cross-validation, SHAP-driven feature selection, uncertainty "
            f"quantification (bootstrap ensembles), and wavelet anti-noise filtering — "
            f"advancing beyond what {w['team']} achieved by coupling geological fidelity "
            f"with modern LangChain RAG and multi-agent orchestration.\n\n"
            f"{WINNERS_CONTEXT}"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "source": "oz_minerals_challenge_winners",
                "team": w["team"],
                "prize": w["prize"],
                "year": w["year"],
                "doc_type": "winner_analysis",
            }
        ))

    # Also write them to disk for inspection
    for doc in docs:
        fname = WINNERS_KB_PATH / f"{doc.metadata['team'].replace(' ', '_')}.txt"
        fname.write_text(doc.page_content, encoding="utf-8")
    log.info(f"Built {len(docs)} winner knowledge documents")
    return docs


# ─────────────────────────────────────────────────────────────────
# IOCG Geological Knowledge Seed Documents
# ─────────────────────────────────────────────────────────────────

IOCG_KNOWLEDGE = [
    {
        "title": "IOCG Mineral System Overview",
        "content": """
Iron Oxide Copper-Gold (IOCG) deposits are a class of hydrothermal mineral systems
characterised by abundant iron oxide (magnetite or hematite), significant copper
and/or gold mineralisation, and spatial association with large-scale structural
corridors, felsic to mafic magmatism, and breccia zones.

Key Mineral System Components (targeting proxies):
1. ENERGY SOURCE: Crustal-scale faults (flower/transpressional), magmatic centres.
   Proxy: TMI gradients, basement depth from gravity inversion.
2. FLUID SOURCE: Magmatic-hydrothermal, metamorphic, or evaporitic brines.
   Proxy: Potassic alteration (K/eTh radiometrics), Na depletion aureoles.
3. FLUID PATHWAY: Second-order faults, fault intersections, dilational jogs.
   Proxy: Fault density maps, lineament intersections, structural complexity index.
4. TRAP/DEPOSITIONAL SITE: Lithological contacts, rheological contrasts, breccia zones.
   Proxy: Geology unit boundaries, gravity-magnetic discontinuities.
5. PRESERVATIONAL POTENTIAL: Cover depth, erosion level.
   Proxy: Isostatic gravity residual, cover thickness models.

Prominent Hill / Mount Woods IOCG Setting:
- Gawler Craton, South Australia
- Olympic IOCG Province (Olympic Dam, Prominent Hill, Carrapateena)
- Hiltaba Suite granites (1590 Ma) as heat/fluid driver
- Gawler Range Volcanics as stratigraphy
- Regolith-covered: deep drilling required
- Key geophysical signature: circular/elliptical negative TMI anomaly with
  associated gravity low (magnetite destruction + low-density breccia body)
"""
    },
    {
        "title": "Geophysical Feature Engineering for IOCG Prospectivity",
        "content": """
Critical geophysical derivatives for IOCG targeting in the Gawler Craton:

MAGNETICS (TMI):
- Total Horizontal Derivative (THD): maps contacts and fault traces
- Analytical Signal (AS): locates sources regardless of remanence
- Upward Continuation (1, 2, 5 km): isolates deep vs shallow sources
- Reduction to Pole (RTP): removes inclination effects at -65° inclination
- Tilt Derivative: excellent for mapping buried intrusions
- 2nd Vertical Derivative: enhances shallow features
- RTP Analytic Signal: combines RTP + AS for robust body edge detection

GRAVITY (Bouguer Anomaly):
- Isostatic Residual Gravity: removes long-wavelength crustal effects
- Horizontal Gradient (gravity edges): maps density contrasts
- Upward Continuation: deep basin/basement structures
- Multi-scale Edge Detection (MEDUSA): automated lineament extraction

RADIOMETRICS (K, eTh, eU, Total Dose):
- Potassium (K%): maps potassic alteration (strong IOCG proxy)
- K/eTh ratio: discriminates potassic alteration from K-rich lithologies
- F-parameter composite: (2K + eTh/4 + eU/2): general radioelement map
- Gamma-ray spectrometry ternary (K=R, eTh=G, eU=B): visual alteration map

INTEGRATION:
- Magnetic susceptibility * gravity density cross-plot → breccia body detection
- Weighted overlay prospectivity index (WofE, logistic regression, or ML)
"""
    },
    {
        "title": "Spatial Cross-Validation for Mineral Prospectivity",
        "content": """
Standard cross-validation overestimates performance for spatial prediction tasks
due to spatial autocorrelation (nearby samples share geology → data leakage).

SPATIAL CROSS-VALIDATION approaches for mineral prospectivity:
1. Block Cross-Validation: divide study area into geographic blocks larger than
   the spatial autocorrelation range; train on all-but-one blocks, test on left-out block.
2. Buffered Leave-One-Out (BLOO): exclude all training samples within buffer distance
   of each test sample (10 km recommended for geophysics, 5 km for geochemistry).
3. Environmental Blocking: cluster samples by geophysical properties; ensure
   train/test differs in feature space as well as geographic space.

Recommended metrics:
- AUC-ROC (spatial CV estimate): primary metric
- Precision-Recall AUC: robust to class imbalance (rare deposits)
- Brier Score: probability calibration assessment
- Reliability diagram: visual calibration check

Anti-overfitting measures:
- SHAP-based feature selection: remove features with near-zero SHAP contribution
- Regularisation (XGBoost: reg_alpha, reg_lambda; RF: min_samples_leaf)
- Early stopping on validation fold
- Model Calibration (Platt scaling / isotonic regression)
- Bootstrap uncertainty: spread of predictions across N bootstrap models
"""
    },
]


def _build_geological_knowledge_docs() -> List[Document]:
    """Create Document objects from IOCG knowledge seed."""
    docs = []
    for entry in IOCG_KNOWLEDGE:
        docs.append(Document(
            page_content=entry["content"].strip(),
            metadata={"source": "iocg_knowledge_base", "title": entry["title"], "doc_type": "geology"}
        ))
    log.info(f"Built {len(docs)} IOCG geological knowledge documents")
    return docs


# ─────────────────────────────────────────────────────────────────
# Document Loaders
# ─────────────────────────────────────────────────────────────────

def load_documents_from_directory(directory: Path, glob: str = "**/*") -> List[Document]:
    """Load all supported documents (PDF, TXT, CSV) from a directory."""
    docs: List[Document] = []
    for pdf in directory.rglob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf))
            docs.extend(loader.load())
            log.info(f"Loaded PDF: {pdf.name}")
        except Exception as e:
            log.warning(f"PDF load failed ({pdf.name}): {e}")
    for txt in directory.rglob("*.txt"):
        try:
            loader = TextLoader(str(txt), encoding="utf-8", autodetect_encoding=True)
            docs.extend(loader.load())
        except Exception as e:
            log.warning(f"TXT load failed ({txt.name}): {e}")
    for csv in directory.rglob("*.csv"):
        try:
            loader = CSVLoader(str(csv))
            docs.extend(loader.load())
        except Exception as e:
            log.warning(f"CSV load failed ({csv.name}): {e}")
    log.info(f"Loaded {len(docs)} documents from {directory}")
    return docs


# ─────────────────────────────────────────────────────────────────
# Vector Store Management
# ─────────────────────────────────────────────────────────────────

class JRVectorStore:
    """Manages the FAISS-backed persistent vector store for JR MineralForge."""

    INDEX_PATH = Path(VECTOR_STORE_PATH) / "faiss_index"

    def __init__(self):
        self.embeddings = get_embeddings()
        self.store: Optional[FAISS] = None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_or_create(self, force_rebuild: bool = False) -> "JRVectorStore":
        """Load existing index or build a new one from all knowledge sources."""
        if self.INDEX_PATH.exists() and not force_rebuild:
            log.info(f"Loading existing FAISS index from {self.INDEX_PATH}")
            self.store = FAISS.load_local(
                str(self.INDEX_PATH),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            log.info(f"Loaded index with {self.store.index.ntotal} vectors")
        else:
            self.rebuild()
        return self

    def rebuild(self) -> None:
        """Build the FAISS index from scratch from all document sources."""
        log.info("Building FAISS vector store from scratch …")
        all_docs: List[Document] = []

        # 1. Winner synthetic knowledge
        all_docs.extend(_build_winner_documents())

        # 2. IOCG geological knowledge
        all_docs.extend(_build_geological_knowledge_docs())

        # 3. Any downloaded reports/documents on disk
        for data_subdir in [DATA_DIR / "reports", DATA_DIR / "raw", WINNERS_KB_PATH]:
            if data_subdir.exists():
                all_docs.extend(load_documents_from_directory(data_subdir))

        # Split into chunks
        chunks = self.splitter.split_documents(all_docs)
        log.info(f"Split into {len(chunks)} chunks")

        # Embed and index
        self.store = FAISS.from_documents(chunks, self.embeddings)
        self.INDEX_PATH.mkdir(parents=True, exist_ok=True)
        self.store.save_local(str(self.INDEX_PATH))
        log.info(f"Saved FAISS index → {self.INDEX_PATH} ({self.store.index.ntotal} vectors)")

    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to the existing index (incremental update)."""
        if self.store is None:
            self.load_or_create()
        chunks = self.splitter.split_documents(documents)
        self.store.add_documents(chunks)  # type: ignore
        self.store.save_local(str(self.INDEX_PATH))  # type: ignore
        log.info(f"Added {len(chunks)} new chunks to index")

    def get_retriever(self, k: int = RETRIEVAL_K, score_threshold: float = RETRIEVAL_SCORE_THRESHOLD):
        """Return a anti-noise retriever that filters by score threshold."""
        if self.store is None:
            self.load_or_create()
        return self.store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold},
        )

    def similarity_search_with_scores(self, query: str, k: int = RETRIEVAL_K):
        """Raw similarity search returning (Document, score) pairs."""
        if self.store is None:
            self.load_or_create()
        return self.store.similarity_search_with_score(query, k=k)  # type: ignore


# ─────────────────────────────────────────────────────────────────
# RAG Chain Builder
# ─────────────────────────────────────────────────────────────────

JR_RAG_SYSTEM_TEMPLATE = """You are the JR MineralForge geological AI assistant for Team JR.
{header}

Use the following retrieved geological knowledge to answer the question.
Always cite your sources when referencing specific winner strategies or geological principles.
If the context does not contain sufficient information, say so clearly rather than fabricating an answer.
If referencing previous competition winners, always frame it as:
'{winners_context}'

Retrieved Context:
{{context}}

Question: {{question}}

Answer (professional, geologically precise, cited):""".format(
    header=BRAND_HEADER,
    winners_context=WINNERS_CONTEXT,
)

JR_RAG_PROMPT = PromptTemplate(
    template=JR_RAG_SYSTEM_TEMPLATE,
    input_variables=["context", "question"],
)


def build_rag_chain(llm, vector_store: Optional[JRVectorStore] = None) -> RetrievalQA:
    """Build and return a RetrievalQA chain using the JR vector store."""
    if vector_store is None:
        vector_store = JRVectorStore().load_or_create()
    retriever = vector_store.get_retriever()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": JR_RAG_PROMPT},
    )
    return chain
