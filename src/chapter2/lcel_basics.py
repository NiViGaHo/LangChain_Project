from src.config import set_environment
set_environment()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

# Direct messages
chat = ChatOpenAI(model="gpt-4o-mini")
resp = chat.invoke([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
])
print("Direct:", resp.content)

# LCEL chain
prompt = ChatPromptTemplate.from_template("Tell me a short fact about {topic}")
llm = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | llm | StrOutputParser()
print("LCEL:", chain.invoke({"topic": "the moon"}))
