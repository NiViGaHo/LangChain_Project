# Chapter 2 — First Steps with LangChain

**Focus:** setup, prompts/templates, LCEL pipelines, local models (Ollama/HF).
- Prompt templates and LCEL basics around Side **37–44**; LCEL is the modern chain abstraction built on Runnables.
- Running local models (Ollama + Hugging Face) around **Side 45–46**.

**Files:**
- `src/chapter2/lcel_basics.py` — minimal LCEL chain with `ChatOpenAI`.
- `src/chapter2/local_ollama.py` — LCEL with `ChatOllama`.
- `src/chapter2/hf_pipeline.py` — `HuggingFacePipeline` string-in → text-out.

**Intent:** Demonstrate LCEL composability and vendor-agnostic design; avoid legacy `LLMChain`.
