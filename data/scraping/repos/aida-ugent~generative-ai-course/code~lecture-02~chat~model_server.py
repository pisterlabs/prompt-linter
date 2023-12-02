"""
API gateway
"""
import argparse
import sys
sys.path.append("../")

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from chat.constants import API_HOST, API_PORT, CHAT_MODEL_REPO_NAME, CHAT_PROMPT_TEMPLATE, CONTEXT_PROMPT_TEMPLATE, MAX_LENGTH

app = FastAPI()

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_REPO_NAME)
    model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL_REPO_NAME,  device_map="auto", torch_dtype=torch.float16, do_sample=True, low_cpu_mem_usage=True)
    chat_pipeline = pipeline(task="text-generation", model=model, max_length=MAX_LENGTH, tokenizer=tokenizer)
    LLM = HuggingFacePipeline(pipeline=chat_pipeline)
    prompt = PromptTemplate(input_variables=["user_msg"], template=CHAT_PROMPT_TEMPLATE)
    llm_chain = LLMChain(llm=LLM, verbose=True, prompt=prompt)
    return llm_chain

@app.post("/generate_stream")
async def generate_stream(request: Request):
    params = await request.json()    
    user_msg = params.get("user_msg")    
    context = params.get("context", None)
    if context is not None:        
        user_msg = CONTEXT_PROMPT_TEMPLATE.format(context_str=context, query_str=user_msg)
    text = llm_chain({"user_msg": user_msg.strip("\n")})["text"].strip()
    return Response(content=text)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()    
    parser.add_argument("--host", type=str, default=API_HOST)
    parser.add_argument("--port", type=int, default=API_PORT)    
    args = parser.parse_args()    

    llm_chain = load_model()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    