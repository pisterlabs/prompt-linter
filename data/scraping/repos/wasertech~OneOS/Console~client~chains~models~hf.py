from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

def get_hf_llm(model_name="wasertech/assistant-phi1_5-4k", max_tokens=500):
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto", device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained("wasertech/assistant-llama2-7b-chat", trust_remote_code=True, torch_dtype="auto", device_map='auto')
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_tokens, device_map='auto', trust_remote_code=True
    )
    hf = HuggingFacePipeline(pipeline=pipe)
    return hf

