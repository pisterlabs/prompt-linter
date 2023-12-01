## for conversation LLM
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import HuggingFacePipeline
# import torch
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import os

# from lib_webLLM import WebLLM

OPTION_CUDA_USE_GPU = os.getenv('OPTION_CUDA_USE_GPU', 'False') == "True"
cache_dir = "./cache"


def getFlanLarge():
    
    model_id = 'google/flan-t5-large'
    print(f">> Prep. Get {model_id} ready to go")
    # model_id = 'google/flan-t5-large'# go for a smaller model if you dont have the VRAM
    tokenizer = AutoTokenizer.from_pretrained(model_id) 
    if OPTION_CUDA_USE_GPU:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir=cache_dir, load_in_8bit=True, device_map='auto') 
            model.cuda()
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir=cache_dir) 
    
    pipe = pipeline(
        "text2text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=100
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

## options are flan and stablelm
MODEL = "flan"
local_llm = getFlanLarge()


def make_the_llm():
    template_informed = """
    I am a helpful AI that answers questions. When I don't know the answer I say I don't know. 
    I know context: {context}
    when asked: {question}
    my response using only information in the context is: """

    prompt_informed = PromptTemplate(template=template_informed, input_variables=["context", "question"])

    return LLMChain(prompt=prompt_informed, llm=local_llm)