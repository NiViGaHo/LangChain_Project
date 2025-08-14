import os
from src.config import set_environment


def build_chain():
    """Build and return a minimal RAG chain.

    Heavy imports are placed inside the function to avoid side effects on import.
    """
    set_environment()
    os.environ.setdefault("USER_AGENT", "generative-ai-with-langchain-akse/0.1")

    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
    from langchain_ollama import ChatOllama
    from langchain_openai import ChatOpenAI

    # 1) Load docs
    loader = WebBaseLoader("https://python.langchain.com/docs/introduction/")
    docs = loader.load()

    # 2) Split
    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)

    # 3) Vectorize + retriever (prefer local embeddings)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vs.as_retriever()

    # 4) RAG chain (prefer local model; fallback to OpenAI if available)
    try:
        # Use local Ollama if running
        llm = ChatOllama(model="llama3", temperature=0)
    except Exception:
        # Lightweight HF model to avoid large downloads
        llm = HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-small",
            task="text2text-generation",
            pipeline_kwargs={"max_new_tokens": 128}
        )

    prompt = ChatPromptTemplate.from_template(
        "Answer ONLY from the context.\n\n{context}\n\nQuestion: {question}"
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    rag = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return rag


def main():
    chain = build_chain()
    print(chain.invoke("What is LCEL?"))


if __name__ == "__main__":
    main()
