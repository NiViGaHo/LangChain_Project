from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline


@dataclass
class RAGConfig:
    source_url: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_generation_model: str = "google/flan-t5-small"
    max_new_tokens: int = 128


class RAGPipeline:
    """Factory for building a minimal RAG pipeline given a config."""

    def __init__(self, config: RAGConfig):
        self.config = config

    @classmethod
    def from_url(cls, url: str, **kwargs) -> "RAGPipeline":
        return cls(RAGConfig(source_url=url, **kwargs))

    def build(self):
        os.environ.setdefault("USER_AGENT", "generative-ai-with-langchain-akse/0.1")
        loader = WebBaseLoader(self.config.source_url)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        splits = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
        vector_store = FAISS.from_documents(splits, embeddings)
        retriever = vector_store.as_retriever()

        llm = HuggingFacePipeline.from_model_id(
            model_id=self.config.hf_generation_model,
            task="text2text-generation",
            pipeline_kwargs={"max_new_tokens": self.config.max_new_tokens},
        )

        prompt = ChatPromptTemplate.from_template(
            "Answer ONLY from the context.\n\n{context}\n\nQuestion: {question}"
        )

        def format_docs(docs: Iterable) -> str:
            return "\n\n".join(d.page_content for d in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain


