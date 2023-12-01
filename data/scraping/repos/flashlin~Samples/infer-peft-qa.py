import json
import os
from pprint import pprint
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from finetune_utils import load_finetune_config
from langchain_lit import load_markdown_documents, LlmEmbedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain

config = load_finetune_config()
device = "cuda"
EMB_MODEL = "BAAI_bge-base-en"

def load_vector_store():
    print("loading data")
    docs = load_markdown_documents("./data")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=35)
    all_splits = text_splitter.split_documents(docs)
    llm_embedding = LlmEmbedding(f"../models/{EMB_MODEL}")
    print("loading vector")
    vectorstore = Chroma.from_documents(documents=all_splits,
                                        embedding=llm_embedding.embedding)
    return vectorstore


def clean_prompt_resp(resp: str):
    after_inst = resp.split("[/INST]", 1)[-1]
    s2 = after_inst.split("[INST]", 1)[0]
    return s2.split('[/INST]', 1)[0]


model_name = config['model_name']
base_model = f"./models/{model_name}"
peft_model_id = f"./outputs/{model_name}-tuned"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    #return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    # trust_remote_code=True,
    local_files_only=True,
)
model.load_adapter(peft_model_id)

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

# model = PeftModel.from_pretrained(model, PEFT_MODEL)

generation_config = model.generation_config
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.1  # 0.7
# generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id


prompt_template = """<s>[INST] {user_input} [/INST]"""


from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
task = "text-generation"
pipe = pipeline(
    task=task,
    model=model, 
    tokenizer=tokenizer, 
    max_length=1024,
    temperature=0.1,
    top_p=0.95,
    repetition_penalty=1.15
)
llm = HuggingFacePipeline(pipeline=pipe)

vectorstore = load_vector_store()
qachain = RetrievalQA.from_chain_type(llm, 
                                        chain_type='stuff', 
                                        retriever=vectorstore.as_retriever(
                                        search_kwargs={'k': 10, 'fetch_k': 50}
                                        ))

def ask(user_input):
     prompt = prompt_template.format(user_input=user_input)
     encoding = tokenizer(prompt, return_tensors="pt").to(device)

     outputs = model.generate(
        input_ids=encoding.input_ids,
        attention_mask=encoding.attention_mask,
        generation_config=generation_config
     )

     resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
     answer = clean_prompt_resp(resp)
     return answer
                                        

def ask_qa(user_input):
   resp = qachain(user_input)
   answer = resp['result'] 
   return answer

print(f"load {model_name} done")
with torch.inference_mode():
    while True:
        user_input = input("query: ")
        if user_input == '/bye':
            break

        answer = ask_qa(user_input)
        print("--------------------------------------------------")
        print(answer)
        print("")
        print("--------------------------------------------------")
