"""
Enhanced RAG implementation with comprehensive documentation and error handling.
This serves as a template for AI agents building similar systems.
"""
import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging

from src.config import set_environment
set_environment()

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ.setdefault("USER_AGENT", "generative-ai-with-langchain-akse/0.1")

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline.
    
    Attributes:
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks to maintain context
        use_openai: Whether to use OpenAI models (requires API key)
        embedding_model: Model name for embeddings
        llm_model: Model name for language model
        temperature: LLM temperature for response generation
    """
    chunk_size: int = 1000
    chunk_overlap: int = 200
    use_openai: bool = False
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "llama3"
    temperature: float = 0


class RAGPipeline:
    """
    Production-ready RAG pipeline with error handling and configuration.
    
    This class demonstrates best practices for building RAG systems:
    - Configurable components
    - Proper error handling
    - Logging for debugging
    - Type hints for clarity
    - Fallback mechanisms
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize RAG pipeline with configuration.
        
        Args:
            config: Optional configuration object. Uses defaults if not provided.
        """
        self.config = config or RAGConfig()
        self.retriever = None
        self.chain = None
        logger.info(f"Initializing RAG pipeline with config: {self.config}")
    
    def load_documents(self, url: str) -> List[Document]:
        """Load documents from a web URL.
        
        Args:
            url: Web URL to load documents from
            
        Returns:
            List of loaded documents
            
        Raises:
            Exception: If document loading fails
        """
        try:
            logger.info(f"Loading documents from: {url}")
            loader = WebBaseLoader(url)
            docs = loader.load()
            logger.info(f"Successfully loaded {len(docs)} documents")
            return docs
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise
    
    def split_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into chunks for processing.
        
        Args:
            docs: List of documents to split
            
        Returns:
            List of document chunks
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        splits = splitter.split_documents(docs)
        logger.info(f"Split into {len(splits)} chunks")
        return splits
    
    def create_retriever(self, splits: List[Document]):
        """Create vector store and retriever from document splits.
        
        Args:
            splits: List of document chunks
            
        Returns:
            Configured retriever
        """
        # Choose embeddings based on configuration
        if self.config.use_openai:
            logger.info("Using OpenAI embeddings")
            embeddings = OpenAIEmbeddings()
        else:
            logger.info(f"Using local embeddings: {self.config.embedding_model}")
            embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model
            )
        
        # Create vector store
        logger.info("Creating FAISS vector store")
        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=embeddings
        )
        
        # Create retriever with configuration
        self.retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}  # Return top 4 most relevant chunks
        )
        return self.retriever
    
    def create_llm(self):
        """Create language model with fallback mechanisms.
        
        Returns:
            Configured language model
        """
        if self.config.use_openai:
            logger.info("Using OpenAI LLM")
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=self.config.temperature
            )
        
        # Try Ollama first
        try:
            logger.info(f"Attempting to use Ollama with model: {self.config.llm_model}")
            llm = ChatOllama(
                model=self.config.llm_model,
                temperature=self.config.temperature
            )
            # Test the connection
            llm.invoke("test")
            logger.info("Successfully connected to Ollama")
            return llm
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            
            # Fallback to HuggingFace
            logger.info("Falling back to HuggingFace model")
            from langchain_huggingface import HuggingFacePipeline
            return HuggingFacePipeline.from_model_id(
                model_id="google/flan-t5-small",
                task="text2text-generation",
                pipeline_kwargs={"max_new_tokens": 128}
            )
    
    def build_chain(self):
        """Build the complete RAG chain.
        
        Returns:
            Configured RAG chain
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized. Call create_retriever first.")
        
        llm = self.create_llm()
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant. Answer the question based only on the provided context.
            If you cannot answer the question from the context, say so clearly.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
        )
        
        def format_docs(docs: List[Document]) -> str:
            """Format documents for inclusion in prompt."""
            return "\n\n".join(d.page_content for d in docs)
        
        # Build LCEL chain
        self.chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        logger.info("RAG chain successfully built")
        return self.chain
    
    def query(self, question: str) -> str:
        """Query the RAG system.
        
        Args:
            question: Question to answer
            
        Returns:
            Answer from the RAG system
            
        Raises:
            ValueError: If chain is not initialized
        """
        if not self.chain:
            raise ValueError("Chain not initialized. Call build_chain first.")
        
        logger.info(f"Processing query: {question}")
        try:
            answer = self.chain.invoke(question)
            logger.info("Query processed successfully")
            return answer
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    @classmethod
    def from_url(cls, url: str, config: Optional[RAGConfig] = None) -> 'RAGPipeline':
        """Factory method to create a complete RAG pipeline from a URL.
        
        Args:
            url: Web URL to load documents from
            config: Optional configuration
            
        Returns:
            Configured and ready-to-use RAG pipeline
            
        Example:
            >>> rag = RAGPipeline.from_url("https://example.com", RAGConfig(use_openai=True))
            >>> answer = rag.query("What is the main topic?")
        """
        pipeline = cls(config)
        docs = pipeline.load_documents(url)
        splits = pipeline.split_documents(docs)
        pipeline.create_retriever(splits)
        pipeline.build_chain()
        return pipeline


def main():
    """Demonstrate RAG pipeline usage with different configurations."""
    
    # Example 1: Local models (default)
    print("=" * 50)
    print("Example 1: Using local models")
    print("=" * 50)
    
    rag_local = RAGPipeline.from_url(
        "https://python.langchain.com/docs/introduction/",
        config=RAGConfig(use_openai=False)
    )
    
    answer = rag_local.query("What is LCEL?")
    print(f"Answer: {answer}\n")
    
    # Example 2: OpenAI models (if API key is set)
    if os.getenv("OPENAI_API_KEY"):
        print("=" * 50)
        print("Example 2: Using OpenAI models")
        print("=" * 50)
        
        rag_openai = RAGPipeline.from_url(
            "https://python.langchain.com/docs/introduction/",
            config=RAGConfig(use_openai=True, temperature=0.3)
        )
        
        answer = rag_openai.query("What are the key features of LangChain?")
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()
