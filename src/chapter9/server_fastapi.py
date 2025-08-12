from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.config import set_environment

class ChatRequest(BaseModel):
    message: str

app = FastAPI(title="Simple LangChain Chat API")
set_environment()
llm = ChatOpenAI(model="gpt-4o-mini")

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    ai = await llm.ainvoke([HumanMessage(content=req.message)])
    return {"response": ai.content}
