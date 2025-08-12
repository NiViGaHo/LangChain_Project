from __future__ import annotations
import os
from typing import Iterable, Optional
import pandas as pd
from sklearn.datasets import load_iris

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, BaseMessage
from src.config import set_environment
set_environment()

ENABLE_DANGEROUS_TOOLS = False
if ENABLE_DANGEROUS_TOOLS:
    from langchain_experimental.tools.python.tool import PythonREPLTool

# Data
_iris = load_iris(as_frame=True)
DF = pd.concat([_iris.data, pd.Series(_iris.target, name="target")], axis=1)
DF.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in DF.columns]

def _ensure_numeric(col: str):
    import pandas as pd
    if col not in DF.columns:
        raise ValueError(f"Unknown column: {col}")
    if not pd.api.types.is_numeric_dtype(DF[col]):
        raise ValueError(f"Column '{col}' is not numeric.")

@tool
def df_columns() -> list[str]:
    """List DataFrame columns."""
    return DF.columns.tolist()

@tool
def df_head(n: int = 5) -> str:
    """First n rows as JSON (records)."""
    n = max(1, min(int(n), 50))
    return DF.head(n).to_json(orient="records")

@tool
def df_summary_stats(columns: Optional[list[str]] = None) -> str:
    """count, mean, std, min, max for numeric cols as JSON."""
    use = columns or DF.select_dtypes("number").columns.tolist()
    for c in use: _ensure_numeric(c)
    desc = DF[use].agg(["count","mean","std","min","max"]).transpose()
    return desc.to_json(orient="table")

@tool
def df_value_counts(column: str, top: int = 10) -> str:
    """Top value counts for a column as JSON."""
    if column not in DF.columns:
        raise ValueError(f"Unknown column: {column}")
    top = max(1, min(int(top), 50))
    vc = DF[column].value_counts().head(top).reset_index()
    vc.columns = [column, "count"]
    return vc.to_json(orient="records")

TOOLS = [df_columns, df_head, df_summary_stats, df_value_counts]
if ENABLE_DANGEROUS_TOOLS:
    TOOLS.append(PythonREPLTool())

SYSTEM_PROMPT = (
    "You are a data assistant. Prefer the DataFrame tools for the dataset. "
    "Use Python REPL only if explicitly necessary; no I/O or network. Be concise."
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_react_agent(model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT)

def _last_ai_text(messages: Iterable[BaseMessage]) -> str:
    ai = [m for m in messages if isinstance(m, AIMessage)]
    if not ai: return ""
    c = ai[-1].content
    if isinstance(c, list):
        return "".join(p.get("text","") for p in c if isinstance(p, dict))
    return str(c)

def demo():
    tests = [
        "What are the column names?",
        "Show me the first 3 rows.",
        "Give mean and std for sepal_length and petal_width.",
        "Which target values occur most often?",
        "Compute the ratio of mean(petal_length) to mean(sepal_width) to 4 decimals.",
    ]
    for q in tests:
        res = agent.invoke({"messages": [{"role": "user", "content": q}]})
        print(f"
Q: {q}
A: {_last_ai_text(res['messages'])}")

if __name__ == "__main__":
    demo()
