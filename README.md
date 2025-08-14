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

## Using the new patterns

- RAG pattern example:
```bash
python -m examples.document_qa.run_rag_pattern
```

- In code:
```python
from src.patterns import RAGPipeline
chain = RAGPipeline.from_url("https://python.langchain.com/docs/introduction/").build()
print(chain.invoke("What is LCEL?"))
```

## Test the API

PowerShell:
```powershell
# If your path contains spaces, prefer the helper scripts below
./scripts/start_api.ps1 -Port 8010
./scripts/call_chat.ps1 -Port 8010 -Message 'What is LCEL?'
```

## Structure

- `src/chapter2`: LCEL basics + running local models (Ollama, HF).
- `src/chapter3`: LangGraph stateful workflow with conditional edges.
- `src/chapter4`: Minimal RAG pipeline (loader → splitter → FAISS → retriever → LCEL) and `patterns.rag_pattern`.
- `src/chapter7`: LangGraph ReAct agent; reusable pattern in `patterns.agent_pattern`.
- `src/chapter9`: Async-safe FastAPI wrapper using the RAG pattern.
- `src/patterns`: Reusable RAG/Agent/Workflow builders.
- `docs`: Architecture and best practices.

## Windows notes (paths with spaces)

- Use `-LiteralPath` in `Set-Location`, or prefer the helper scripts:
  - `scripts/start_api.ps1` runs Uvicorn using the venv python path safely.
  - `scripts/call_chat.ps1` posts to the API with proper quoting.

## Page mapping (broken Index)

The book's Index pages are ~**+3** ahead of the actual PDF "Side N". Use: `pdf_side ≈ index_page − 3` (±1). See `docs/MAPPING.md` for quick references.
