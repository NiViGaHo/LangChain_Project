# Generative AI with LangChain — runnable examples (akse.ai edition)

Production-minded examples aligned to *Generative AI with LangChain – Second Edition*, updated for LangChain 0.3.x and LangGraph 0.2.x.

## Quickstart

```bash
# 1) create an isolated env
#    - Unix/macOS (conda):
bash scripts/conda_setup.sh
#    - Windows PowerShell (venv):
./scripts/venv_setup.ps1

# 2) set your keys
#    - Unix/macOS:
cp .env.example .env
#    - Windows PowerShell:
Copy-Item .env.example .env
# edit .env with your OPENAI_API_KEY etc.

# 3) run samples (module form ensures imports work)
python -m src.chapter2.lcel_basics
python -m src.chapter3.langgraph_workflow
python -m src.chapter4.rag_minimal
python -m src.chapter7.agent_langgraph_pandas_python
uvicorn src.chapter9.server_fastapi:app --reload
```

## Why these versions

* LangChain 0.3 split partner packages and favors LCEL/Runnables; agents are steered toward LangGraph.
* Chapter coverage: LCEL appears around **Side 39–44** (Index: 42–44). LangGraph starts **Side 61**. RAG core concepts around **Side 92–110+**. Use our page mapping guide.

## Structure

* `src/chapter2`: LCEL basics + running local models (Ollama, HF).
* `src/chapter3`: LangGraph stateful workflow with conditional edges.
* `src/chapter4`: Minimal RAG pipeline (loader → splitter → FAISS → retriever → LCEL).
* `src/chapter7`: LangGraph ReAct agent with safe Pandas tools (+ optional Python REPL).
* `src/chapter9`: Async-safe FastAPI wrapper.

## Page mapping (broken Index)

The book's Index pages are ~**+3** ahead of the actual PDF "Side N". Use: `pdf_side ≈ index_page − 3` (±1). See `docs/MAPPING.md` for quick references.
