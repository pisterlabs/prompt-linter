from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
from huggingface_hub.hf_api import HfFolder
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
from llms.llm import Llm

CACHE_CATALOG = os.getcwd() + "/.model_cache"

class HuggingFace(Llm):
    def __init__(self):
        load_dotenv()
        hf_api_key = os.getenv("HF_API_KEY")
        HfFolder.save_token(hf_api_key)

    def get_llm(self):

        model = os.getenv("MODEL_ID")

        tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=CACHE_CATALOG)

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            max_length=1000,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )

        return HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0, 'cache_dir': CACHE_CATALOG})

