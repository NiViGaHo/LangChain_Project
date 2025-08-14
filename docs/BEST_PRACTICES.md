# LangChain Best Practices

## 1. Type Hints Everywhere
Use explicit type hints for functions, classes, and external interfaces.

## 2. Proper Error Handling
Catch known errors (rate limits, network) and log unexpected exceptions.

## 3. Configuration Objects
Prefer dataclasses for configuration. Avoid long positional argument lists.

## 4. Observability
- Structured logging (see `src/core/logging_config.py`)
- Metrics and tracing (optional LangSmith)

## 5. Design for Testability
- Use dependency injection
- Mock external services in unit tests
- Add integration tests for critical flows

## 6. Modern LangChain Patterns
- Prefer LCEL and LangGraph over legacy chains
- Use split partner packages (e.g., `langchain_openai`, `langchain_community`)

## 7. Local-First Development
- Prefer local embeddings and small HF models for prototyping
- Gate cloud usage behind configuration

## 8. Patterns in Practice
- RAG: use `RAGConfig` + `RAGPipeline.from_url(...).build()`
- Agent: build with `AgentConfig` + `build_pandas_agent`
- Workflow: assemble graphs via `workflow_pattern.build_safety_workflow()`

## 9. API Surfaces
- Use `.ainvoke` in async servers
- Keep chain construction outside request handlers where possible
