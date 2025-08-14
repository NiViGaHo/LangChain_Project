# Architecture Guide for AI Agents

## Design Patterns Used

### 1. Factory Pattern (RAGPipeline.from_url)
- Why: Simplifies complex object creation
- When to use: When multiple steps are needed to initialize a pipeline

### 2. Configuration Pattern (RAGConfig dataclass)
- Why: Centralizes parameters and makes code reusable
- When to use: When many parameters might change per environment

### 3. Chain of Responsibility (LCEL)
- Why: Enables modular processing pipelines
- When to use: When data flows through multiple processing steps

## Component Selection Guide

| Use Case | Recommended Components | Fallback Options |
|----------|------------------------|------------------|
| Fast local inference | Ollama + llama3 | HuggingFace Flan-T5 |
| Production quality | OpenAI GPT-4 | Anthropic Claude |
| Embeddings (local) | sentence-transformers | HuggingFace models |
| Vector Store (simple) | FAISS | Chroma |
| Vector Store (production) | Pinecone/Weaviate | PostgreSQL + pgvector |

## Notes
- Prefer LCEL and LangGraph for modern LangChain code.
- Keep `.env` for local development; prefer secret managers in production.
- All examples are designed to be runnable with minimal setup.

## Chapter Implementations

- Chapter 3 (Workflow): implemented as a reusable safety workflow in `src/patterns/workflow_pattern.py` and a concrete example in `src/chapter3/langgraph_workflow.py`.
- Chapter 4 (RAG): exposed as a pattern in `src/patterns/rag_pattern.py` and used in the API (`chapter9`) and example runner.
- Chapter 7 (Agent): refactored into `src/patterns/agent_pattern.py` and used by the chapter demo.
- Chapter 9 (API): uses the RAG pattern and `ainvoke` for async-safe inference.
