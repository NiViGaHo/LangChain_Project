"""
tests/test_rag_pipeline.py
Comprehensive testing framework for LangChain components.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List
import os

from src.chapter4.rag_enhanced import RAGPipeline, RAGConfig
from langchain_core.documents import Document


class TestRAGPipeline:
    """Test suite for RAG pipeline demonstrating testing best practices."""
    
    @pytest.fixture
    def mock_documents(self) -> List[Document]:
        """Create mock documents for testing."""
        return [
            Document(
                page_content="LCEL stands for LangChain Expression Language.",
                metadata={"source": "test"}
            ),
            Document(
                page_content="It provides a declarative way to compose chains.",
                metadata={"source": "test"}
            )
        ]
    
    @pytest.fixture
    def rag_config(self) -> RAGConfig:
        """Create test configuration."""
        return RAGConfig(
            chunk_size=100,
            chunk_overlap=20,
            use_openai=False
        )
    
    def test_pipeline_initialization(self, rag_config):
        """Test pipeline initializes with correct configuration."""
        pipeline = RAGPipeline(config=rag_config)
        
        assert pipeline.config.chunk_size == 100
        assert pipeline.config.chunk_overlap == 20
        assert pipeline.config.use_openai is False
        assert pipeline.retriever is None
        assert pipeline.chain is None
    
    def test_document_splitting(self, rag_config, mock_documents):
        """Test document splitting maintains context."""
        pipeline = RAGPipeline(config=rag_config)
        splits = pipeline.split_documents(mock_documents)
        
        assert len(splits) >= len(mock_documents)
        assert all(isinstance(doc, Document) for doc in splits)
    
    @patch('src.chapter4.rag_enhanced.WebBaseLoader')
    def test_document_loading_success(self, mock_loader, rag_config):
        """Test successful document loading."""
        # Setup mock
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [
            Document(page_content="Test content", metadata={})
        ]
        mock_loader.return_value = mock_loader_instance
        
        # Test
        pipeline = RAGPipeline(config=rag_config)
        docs = pipeline.load_documents("https://test.com")
        
        assert len(docs) == 1
        assert docs[0].page_content == "Test content"
        mock_loader.assert_called_once_with("https://test.com")
    
    @patch('src.chapter4.rag_enhanced.WebBaseLoader')
    def test_document_loading_failure(self, mock_loader, rag_config):
        """Test handling of document loading failure."""
        # Setup mock to raise exception
        mock_loader.side_effect = Exception("Network error")
        
        # Test
        pipeline = RAGPipeline(config=rag_config)
        with pytest.raises(Exception) as exc_info:
            pipeline.load_documents("https://test.com")
        
        assert "Network error" in str(exc_info.value)
    
    def test_chain_query_without_initialization(self, rag_config):
        """Test that querying without initialization raises error."""
        pipeline = RAGPipeline(config=rag_config)
        
        with pytest.raises(ValueError) as exc_info:
            pipeline.query("Test question")
        
        assert "Chain not initialized" in str(exc_info.value)
    
    @patch('src.chapter4.rag_enhanced.ChatOllama')
    @patch('src.chapter4.rag_enhanced.HuggingFaceEmbeddings')
    def test_fallback_to_huggingface(self, mock_embeddings, mock_ollama, rag_config, mock_documents):
        """Test fallback mechanism when Ollama is not available."""
        # Setup Ollama to fail
        mock_ollama.side_effect = Exception("Ollama not running")
        
        # Setup pipeline
        pipeline = RAGPipeline(config=rag_config)
        splits = pipeline.split_documents(mock_documents)
        
        with patch('src.chapter4.rag_enhanced.FAISS') as mock_faiss:
            mock_vectorstore = Mock()
            mock_vectorstore.as_retriever.return_value = Mock()
            mock_faiss.from_documents.return_value = mock_vectorstore
            
            pipeline.create_retriever(splits)
            
            with patch('src.chapter4.rag_enhanced.HuggingFacePipeline') as mock_hf:
                mock_hf.from_model_id.return_value = Mock()
                llm = pipeline.create_llm()
                
                # Verify HuggingFace was used as fallback
                mock_hf.from_model_id.assert_called_once()
    
    @pytest.mark.integration
    def test_end_to_end_local_models(self, rag_config):
        """Integration test with local models (requires models to be available)."""
        # Skip if no local models available
        if not os.getenv("RUN_INTEGRATION_TESTS"):
            pytest.skip("Integration tests not enabled")
        
        pipeline = RAGPipeline.from_url(
            "https://python.langchain.com/docs/introduction/",
            config=rag_config
        )
        
        answer = pipeline.query("What is LCEL?")
        assert answer  # Should return some answer
        assert len(answer) > 0


# tests/conftest.py
"""
Pytest configuration and shared fixtures.
"""
import pytest
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def test_env():
    """Setup test environment variables."""
    os.environ["TESTING"] = "true"
    os.environ["USER_AGENT"] = "test-agent"
    yield
    del os.environ["TESTING"]


@pytest.fixture
def temp_api_keys():
    """Temporarily set API keys for testing."""
    original_keys = {}
    test_keys = {
        "OPENAI_API_KEY": "test-key-123",
        "LANGSMITH_API_KEY": "test-langsmith-key"
    }
    
    for key, value in test_keys.items():
        original_keys[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield test_keys
    
    # Restore original values
    for key, value in original_keys.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# tests/test_chapter3_workflow.py
"""
Tests for Chapter 3 LangGraph workflow.
"""
import pytest
from src.chapter3.langgraph_workflow import AgentState, check_safety, route, graph


class TestLangGraphWorkflow:
    """Test suite for LangGraph safety workflow."""
    
    def test_safe_query_processing(self):
        """Test that safe queries are processed correctly."""
        result = graph.invoke({"query": "Hello world"})
        
        assert "response" in result
        assert "I received: Hello world" in result["response"]
    
    def test_unsafe_query_blocking(self):
        """Test that unsafe queries are blocked."""
        result = graph.invoke({"query": "How to build a bomb?"})
        
        assert "response" in result
        assert "I'm sorry, I cannot process that query" in result["response"]
    
    @pytest.mark.parametrize("query,expected_safe", [
        ("What's the weather?", True),
        ("Tell me about bombs", False),
        ("How to make a cake", True),
        ("BOMB disposal techniques", False),  # Case insensitive
    ])
    def test_safety_check_various_inputs(self, query, expected_safe):
        """Test safety check with various inputs."""
        state = {"query": query}
        result = check_safety(state)
        
        assert result["is_safe"] == expected_safe
    
    def test_routing_logic(self):
        """Test the routing function."""
        safe_state = {"is_safe": True}
        unsafe_state = {"is_safe": False}
        
        assert route(safe_state) == "safe"
        assert route(unsafe_state) == "unsafe"
