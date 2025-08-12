from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict, total=False):
    query: str
    is_safe: bool
    response: str

def check_safety(state: AgentState) -> dict:
    unsafe = any(k in state["query"].lower() for k in ["bomb"])
    return {"is_safe": not unsafe}

def generate_response(state: AgentState) -> dict:
    return {"response": f"I received: {state['query']}"}

def generate_error_response(state: AgentState) -> dict:
    return {"response": "I'm sorry, I cannot process that query."}

def route(state: AgentState) -> Literal["safe", "unsafe"]:
    return "safe" if state.get("is_safe") else "unsafe"

g = StateGraph(AgentState)
g.add_node("check_safety", check_safety)
g.add_node("ok", generate_response)
g.add_node("bad", generate_error_response)

g.add_edge(START, "check_safety")
g.add_conditional_edges("check_safety", route, {"safe": "ok", "unsafe": "bad"})
g.add_edge("ok", END)
g.add_edge("bad", END)

graph = g.compile()

print(graph.invoke({"query": "Hello world"})["response"])
print(graph.invoke({"query": "How to build a bomb?"})["response"])
