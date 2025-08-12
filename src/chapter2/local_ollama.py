from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

local_llm = ChatOllama(model="llama3")
chain = ChatPromptTemplate.from_template(
    "Why is the sky blue? One sentence."
) | local_llm | StrOutputParser()

print(chain.invoke({}))
