"""
Tests – Data Ingestion Agent
Tests for SARIG and GA ingestion pipelines, HTTP helpers, and raster processing.
Mocks all network calls so tests run offline.
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dirs(tmp_path):
    """Provide temporary raw and processed directories."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    return {"root": tmp_path, "raw": raw, "processed": processed}


@pytest.fixture
def mock_zip_bytes():
    """Return bytes of a minimal ZIP file containing a dummy shapefile set."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("surface_geology.shp", b"DUMMY_SHP_CONTENT")
        z.writestr("surface_geology.dbf", b"DUMMY_DBF_CONTENT")
        z.writestr("surface_geology.prj", b"DUMMY_PRJ_CONTENT")
    return buf.getvalue()


@pytest.fixture
def mock_geojson_bytes():
    """Return bytes of a minimal GeoJSON FeatureCollection."""
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [135.5, -29.3]},
                "properties": {"name": "Test Occurrence", "commodity": "Cu-Au"},
            }
        ],
    }
    return json.dumps(fc).encode()


# ─────────────────────────────────────────────────────────────────
# HTTP Download Helper
# ─────────────────────────────────────────────────────────────────

class TestDownloadWithRetry:
    def test_successful_download(self, tmp_path, mock_zip_bytes):
        from agents.data_ingestion_agent import _download_with_retry
        dest = tmp_path / "test.zip"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_content = MagicMock(return_value=[mock_zip_bytes])

        with patch("requests.Session.get", return_value=mock_resp):
            result = _download_with_retry("http://fake.url/test.zip", dest)

        assert result == dest
        assert dest.exists()
        assert dest.stat().st_size > 0

    def test_retry_on_failure_then_success(self, tmp_path, mock_zip_bytes):
        from agents.data_ingestion_agent import _download_with_retry
        import requests
        dest = tmp_path / "retry_test.zip"

        # Fail twice, succeed on 3rd
        mock_fail = MagicMock()
        mock_fail.raise_for_status.side_effect = requests.HTTPError("500")
        mock_ok = MagicMock()
        mock_ok.raise_for_status = MagicMock()
        mock_ok.iter_content = MagicMock(return_value=[mock_zip_bytes])

        with patch("requests.Session.get", side_effect=[mock_fail, mock_fail, mock_ok]):
            with patch("time.sleep"):  # Skip real sleep
                result = _download_with_retry("http://fake.url/retry.zip", dest)
        assert dest.exists()

    def test_all_retries_exhausted_raises(self, tmp_path):
        from agents.data_ingestion_agent import _download_with_retry
        import requests
        dest = tmp_path / "fail.zip"

        mock_fail = MagicMock()
        mock_fail.raise_for_status.side_effect = requests.HTTPError("503")

        with patch("requests.Session.get", return_value=mock_fail):
            with patch("time.sleep"):
                with pytest.raises(RuntimeError, match="Failed to download"):
                    _download_with_retry("http://fake.url/fail.zip", dest)


# ─────────────────────────────────────────────────────────────────
# ZIP Extraction
# ─────────────────────────────────────────────────────────────────

class TestExtractZip:
    def test_extracts_all_files(self, tmp_path, mock_zip_bytes):
        from agents.data_ingestion_agent import _extract_zip
        zip_path = tmp_path / "test.zip"
        zip_path.write_bytes(mock_zip_bytes)
        dest_dir = tmp_path / "extracted"
        dest_dir.mkdir()

        files = _extract_zip(zip_path, dest_dir)
        assert len(files) == 3
        names = {f.name for f in files}
        assert "surface_geology.shp" in names


# ─────────────────────────────────────────────────────────────────
# SARIG Ingestion Agent
# ─────────────────────────────────────────────────────────────────

class TestSARIGIngestionAgent:
    def test_init_creates_directories(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.settings.RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr("config.settings.PROCESSED_DIR", tmp_path / "processed")
        from agents.data_ingestion_agent import SARIGIngestionAgent
        from config.settings import MOUNT_WOODS_BBOX
        agent = SARIGIngestionAgent(bbox=MOUNT_WOODS_BBOX)
        assert agent.raw_dir.exists()
        assert agent.processed_dir.exists()

    def test_download_package_uses_cache(self, tmp_path, mock_zip_bytes, monkeypatch):
        """If ZIP already exists, should not re-download."""
        monkeypatch.setattr("config.settings.RAW_DIR", tmp_path)
        from agents.data_ingestion_agent import SARIGIngestionAgent
        from config.settings import MOUNT_WOODS_BBOX

        agent = SARIGIngestionAgent(bbox=MOUNT_WOODS_BBOX)
        # Pre-create the cache file
        zip_path = agent.raw_dir / "surface_geology.zip"
        zip_path.write_bytes(mock_zip_bytes)

        with patch("agents.data_ingestion_agent._download_with_retry") as mock_dl:
            agent.download_package("surface_geology")
            mock_dl.assert_not_called()

    def test_ingest_all_returns_dict(self, tmp_path, mock_zip_bytes, mock_geojson_bytes, monkeypatch):
        monkeypatch.setattr("config.settings.RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr("config.settings.PROCESSED_DIR", tmp_path / "processed")

        def fake_download(url, dest, session=None, params=None):
            if str(dest).endswith(".zip"):
                dest.write_bytes(mock_zip_bytes)
            else:
                dest.write_bytes(mock_geojson_bytes)
            return dest

        with patch("agents.data_ingestion_agent._download_with_retry", side_effect=fake_download):
            from agents.data_ingestion_agent import SARIGIngestionAgent
            from config.settings import MOUNT_WOODS_BBOX
            agent = SARIGIngestionAgent(bbox=MOUNT_WOODS_BBOX)
            results = agent.ingest_all()

        assert isinstance(results, dict)
        assert "surface_geology" in results or "mineral_occurrences" in results


# ─────────────────────────────────────────────────────────────────
# GA Ingestion Agent
# ─────────────────────────────────────────────────────────────────

class TestGAIngestionAgent:
    def test_init(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.settings.RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr("config.settings.PROCESSED_DIR", tmp_path / "processed")
        from agents.data_ingestion_agent import GAIngestionAgent
        from config.settings import MOUNT_WOODS_BBOX
        agent = GAIngestionAgent(bbox=MOUNT_WOODS_BBOX)
        assert agent.raw_dir.exists()

    def test_download_grid_uses_cache(self, tmp_path, mock_zip_bytes, monkeypatch):
        monkeypatch.setattr("config.settings.RAW_DIR", tmp_path)
        from agents.data_ingestion_agent import GAIngestionAgent
        from config.settings import MOUNT_WOODS_BBOX
        agent = GAIngestionAgent(bbox=MOUNT_WOODS_BBOX)

        zip_path = agent.raw_dir / "magnetics_tmi.zip"
        zip_path.write_bytes(mock_zip_bytes)

        with patch("agents.data_ingestion_agent._download_with_retry") as mock_dl:
            agent.download_grid("magnetics_tmi")
            mock_dl.assert_not_called()

    def test_check_data_availability_tool(self):
        from agents.data_ingestion_agent import check_data_availability
        result = check_data_availability.invoke({})
        assert "Data Availability" in result or "JR MineralForge" in result


# ─────────────────────────────────────────────────────────────────
# Bounding Box Helpers
# ─────────────────────────────────────────────────────────────────

class TestBoundingBoxHelpers:
    def test_bbox_as_tuple(self):
        from config.settings import MOUNT_WOODS_BBOX
        tup = MOUNT_WOODS_BBOX.as_tuple
        assert len(tup) == 4
        assert tup[0] < tup[2]  # min_lon < max_lon
        assert tup[1] < tup[3]  # min_lat < max_lat

    def test_bbox_centre(self):
        from config.settings import MOUNT_WOODS_BBOX
        lat, lon = MOUNT_WOODS_BBOX.centre
        assert -30.0 < lat < -28.0
        assert 134.5 < lon < 136.5

    def test_bbox_to_geodataframe(self):
        from config.settings import MOUNT_WOODS_BBOX
        from utils.geospatial_utils import bbox_to_geodataframe
        gdf = bbox_to_geodataframe(MOUNT_WOODS_BBOX)
        assert len(gdf) == 1
        assert gdf.crs.to_epsg() == 4326
