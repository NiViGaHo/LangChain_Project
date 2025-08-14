from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd
from sklearn.datasets import load_iris

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


@dataclass
class AgentConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    enable_python_repl: bool = False


def _load_iris_df() -> pd.DataFrame:
    data = load_iris(as_frame=True)
    df = pd.concat([data.data, pd.Series(data.target, name="target")], axis=1)
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
    return df


DF = _load_iris_df()


def _ensure_numeric(col: str) -> None:
    import pandas as pd as _pd
    if col not in DF.columns:
        raise ValueError(f"Unknown column: {col}")
    if not _pd.api.types.is_numeric_dtype(DF[col]):
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
    for c in use:
        _ensure_numeric(c)
    desc = DF[use].agg(["count", "mean", "std", "min", "max"]).transpose()
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


def build_pandas_agent(config: AgentConfig = AgentConfig()):
    tools = [df_columns, df_head, df_summary_stats, df_value_counts]
    if config.enable_python_repl:
        from langchain_experimental.tools.python.tool import PythonREPLTool

        tools.append(PythonREPLTool())

    system_prompt = (
        "You are a data assistant. Prefer the DataFrame tools for the dataset. "
        "Use Python REPL only if explicitly necessary; no I/O or network. Be concise."
    )

    llm = ChatOpenAI(model=config.model, temperature=config.temperature)
    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
    return agent


def last_ai_text(messages: Iterable[BaseMessage]) -> str:
    ai = [m for m in messages if isinstance(m, AIMessage)]
    if not ai:
        return ""
    content = ai[-1].content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


