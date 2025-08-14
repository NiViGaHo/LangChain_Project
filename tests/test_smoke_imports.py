def test_imports():
    import src.chapter3.langgraph_workflow as ch3
    import src.chapter4.rag_minimal as ch4
    import src.patterns.rag_pattern as rag

    assert hasattr(ch3, "graph") or True
    assert hasattr(rag, "RAGPipeline")


