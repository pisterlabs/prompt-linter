from langchain.llms import HuggingFacePipeline
from transformers import pipeline


def get_answer(model_id, prompt:str) -> str:
    pipe = pipeline(
        "text2text-generation",
        model=model_id,
        tokenizer=model_id,
        max_length=100
    )
    
    local_llm = HuggingFacePipeline(pipeline=pipe)
    answer = local_llm(prompt)
    return answer