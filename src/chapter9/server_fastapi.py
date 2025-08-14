from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from src.config import set_environment
from src.patterns.rag_pattern import RAGPipeline

class ChatRequest(BaseModel):
    message: str

app = FastAPI(title="Simple LangChain Chat API")
set_environment()
rag_chain = RAGPipeline.from_url("https://python.langchain.com/docs/introduction/").build()

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    # For demo simplicity, use RAG to answer
    # LCEL chains support .ainvoke across components
    answer = await rag_chain.ainvoke(req.message)
    return {"response": answer}
