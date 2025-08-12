import os
from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from src.chapter7.agent_langgraph_pandas_python import agent, _last_ai_text

client = Client()

ds = client.create_dataset(
    dataset_name="pandas-agent-sanity",
    description="Tiny sanity checks for the LangGraph Pandas agent.",
)
examples = [
    {"inputs": {"question": "List the columns."},
     "outputs": {"answer": "sepal_length, sepal_width, petal_length, petal_width, target"}},
    {"inputs": {"question": "How many rows are in the dataset?"},
     "outputs": {"answer": "150"}},
]
client.create_examples(dataset_id=ds.id, examples=examples)

def target(inputs: dict) -> dict:
    res = agent.invoke({"messages": [{"role": "user", "content": inputs["question"]}]})
    return {"answer": _last_ai_text(res["messages"]).strip()}

def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="openai:o3-mini",
        feedback_key="correctness",
    )
    return evaluator(inputs=inputs, outputs=outputs, reference_outputs=reference_outputs)

exp = client.evaluate(
    target,
    data="pandas-agent-sanity",
    evaluators=[correctness_evaluator],
    experiment_prefix="pandas-langgraph-react",
    max_concurrency=2,
)
print(f"LangSmith experiment URL: {exp.url}")
