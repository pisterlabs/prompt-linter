import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from typing import Set
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from streamlit_chat import message
from typing import Any, List, Dict
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import textwrap
from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory


def get_file_path(uploaded_file):
    cwd = os.getcwd()
    temp_dir = os.path.join(cwd, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

f = st.file_uploader("Upload a file", type=(["pdf"]))
if f is not None:
    path_in = get_file_path(f)
    print("*"*10,path_in)
else:
    path_in = None
   
st.header("LangChain - PDF reader and answering Bot")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "model" not in st.session_state:
    # path=r"/home/drmohammad/data/llm/falcon-7b-instruct"
    # path=r"/home/drmohammad/data/llm/Llama-2-7b"
    path=r'/home/drmohammad/Documents/LLM/Llamav2hf/Llama-2-7b-chat-hf'
    tokenizer = AutoTokenizer.from_pretrained(path,
                                          use_auth_token=True,)

    model = AutoModelForCausalLM.from_pretrained(path,
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             use_auth_token=True,
                                            #  load_in_8bit=True,
                                             load_in_4bit=True
                                             )

    pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = 512,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )
    
    llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0})
    st.session_state["model"] = llm

if "vectorstore" not in st.session_state and path_in:
    loader=PyPDFLoader(file_path=path_in)
    documents=loader.load()
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=30,separator="\n")
    docs=text_splitter.split_documents(documents=documents)
    # embeddings=OpenAIEmbeddings()
    model_name = "sentence-transformers/all-mpnet-base-v2"
    hf = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore=FAISS.from_documents(docs,hf)
    vectorstore.save_local('langchain_pyloader/vectorize')
    new_vectorstore=FAISS.load_local("langchain_pyloader/vectorize",hf)
    print("pdf read done and vectorize")
     
    st.session_state["vectorstore"] = new_vectorstore


if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""



def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text

def remove_substring(string, substring):
    return string.replace(substring, "")



def generate(text):
    prompt = get_prompt(text)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs,
                                 max_new_tokens=512,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.eos_token_id,
                                 )
        final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        final_outputs = cut_off_text(final_outputs, '</s>')
        final_outputs = remove_substring(final_outputs, prompt)

    return final_outputs

def parse_text(text):
        wrapped_text = textwrap.fill(text, width=100)
        print(wrapped_text +'\n\n')
    
def pdf_chat(path):
    # print("pdf read done")
    pass
def ans_ret(query,new_vectorstore,chat_history):
    # qa=RetrievalQA.from_chain_type(llm=OpenAI(),chain_type='stuff',retriever=new_vectorstore.as_retriever())
    # chat=ChatOpenAI(verbose=True,temperature=0,openai_api_key="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")
    
    
    # instruction = "context:\n\n{context} \n\nQuestion: {question} \n\n Answer:"
    # system_prompt = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."

    # template = get_prompt(instruction, system_prompt)
    # prompt = PromptTemplate(
    # input_variables=["context", "question"], template=template
    # )
    # chain_type_kwargs = {"prompt": prompt}
    llm=st.session_state["model"]
    
    qa = ConversationalRetrievalChain.from_llm(
       llm=llm, retriever=new_vectorstore.as_retriever()
    )
    res=qa({"question": query, "chat_history":chat_history})
    # print(f"ans return done: {res}")   
    return res



if prompt and path_in:
    
    with st.spinner("Generating response.."):
        
        new_vectorstore=st.session_state["vectorstore"]
        generated_response = ans_ret(
            query=prompt,new_vectorstore=new_vectorstore,chat_history=st.session_state["chat_history"]
        )
    
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(generated_response["answer"])
        st.session_state["chat_history"].append((prompt,generated_response["answer"]))
        
if st.session_state["chat_answers_history"]:
    for gresponse, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(user_query, is_user=True)
        message(gresponse)