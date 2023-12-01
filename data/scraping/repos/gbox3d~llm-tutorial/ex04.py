#%%
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv
import time

# .env 파일 로드
load_dotenv()

#%%
# Hugging Face 파이프라인 생성
pipe = pipeline("text-generation", model=, tokenizer=model_id, max_new_tokens=50)
