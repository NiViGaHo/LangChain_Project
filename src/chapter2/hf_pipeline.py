# Hugging Face local pipeline (string in → text out)
from langchain_huggingface import HuggingFacePipeline

hf = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100}
)

print(hf.invoke("Explain photosynthesis simply."))
