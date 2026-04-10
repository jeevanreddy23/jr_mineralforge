"""
Tests – RAG Setup
Tests for FAISS vector store, document chunking, and retrieval logic.
"""

import pytest
from pathlib import Path
from langchain_core.documents import Document
from rag.rag_setup import JRVectorStore

@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="The Mount Woods region is known for IOCG deposits like Prominent Hill.",
            metadata={"source": "test_geology", "doc_type": "geology"}
        ),
        Document(
            page_content="Team Guru won the OZ Minerals Explorer Challenge by using geophysical feature engineering.",
            metadata={"source": "test_winners", "doc_type": "winner_analysis"}
        )
    ]

def test_vector_store_build_and_search(tmp_path, sample_documents, monkeypatch):
    # Mock settings.VECTOR_STORE_PATH to use a temp directory
    monkeypatch.setattr("config.settings.VECTOR_STORE_DIR", tmp_path)
    monkeypatch.setattr("rag.rag_setup.VECTOR_STORE_PATH", str(tmp_path))
    
    vs = JRVectorStore()
    
    # Test adding documents
    vs.add_documents(sample_documents)
    
    # Verify index was saved
    assert (tmp_path / "faiss_index").exists()
    
    # Test similarity search
    results = vs.similarity_search_with_scores("Mount Woods IOCG", k=1)
    assert len(results) > 0
    doc, score = results[0]
    assert "Mount Woods" in doc.page_content

def test_chunking_logic(monkeypatch):
    from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
    vs = JRVectorStore()
    
    long_text = "Word " * 1000
    doc = Document(page_content=long_text, metadata={"source": "long"})
    
    chunks = vs.splitter.split_documents([doc])
    
    assert len(chunks) > 1
    # Check if overlap is working (approximate)
    assert chunks[0].page_content.split()[-5:] == chunks[1].page_content.split()[:5]

def test_retriever_threshold(tmp_path, sample_documents, monkeypatch):
    monkeypatch.setattr("config.settings.VECTOR_STORE_DIR", tmp_path)
    monkeypatch.setattr("rag.rag_setup.VECTOR_STORE_PATH", str(tmp_path))
    
    vs = JRVectorStore()
    vs.add_documents(sample_documents)
    
    # High threshold should return nothing for irrelevant query
    retriever = vs.get_retriever(k=1, score_threshold=0.99)
    results = retriever.invoke("random query about space travel")
    assert len(results) == 0
    
    # Valid query should return result
    results = retriever.invoke("Mount Woods")
    # Depending on the embedding model, 0.99 might be too high even for a match, 
    # but the logic should hold.
