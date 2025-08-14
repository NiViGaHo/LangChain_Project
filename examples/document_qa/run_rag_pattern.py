from src.patterns.rag_pattern import RAGPipeline


def main() -> None:
    chain = RAGPipeline.from_url("https://python.langchain.com/docs/introduction/").build()
    print(chain.invoke("What is LCEL?"))


if __name__ == "__main__":
    main()


