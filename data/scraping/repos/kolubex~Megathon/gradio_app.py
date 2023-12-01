import gradio as gr
from gradio.components import Textbox

# IMPORTS
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import pandas as pd
from operator import itemgetter
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader

# Embed and store splits
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain import hub
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel
import torch
from langchain.llms import HuggingFaceHub
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
import time
 # RAG chain
from langchain.schema.runnable import RunnablePassthrough

# torch.cuda.set_device('cpu')
# dont use cuda
device = torch.device('cpu')


def load_model():
    embeddings_model_name = "alibidaran/medical_transcription_generator"
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    vectorstore = Chroma(persist_directory="./vectorstore_train/", embedding_function=embeddings)
    retriever = vectorstore.as_retriever()
    rag_prompt = hub.pull("rlm/rag-prompt")
    model_id = 'google/flan-t5-small'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=False, device_map='cpu')

    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=128
    )

    hf = HuggingFacePipeline(pipeline=hf_pipeline)
    from langchain.schema.runnable import RunnablePassthrough
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | hf
    )
    return rag_chain


def model_function(prompt, rag_chain):
    start = time.time()
    print("prompt: ", prompt)
    response = rag_chain.invoke(prompt)
    response += "\n" + "Time taken: " + str(time.time() - start) + " seconds"
    return (response)


if __name__ == "__main__":
    rag_chain = load_model()
    print("model loaded")
    # Define the interface
    interface = gr.Interface(
        fn=lambda prompt: model_function(prompt, rag_chain),  # function to call
        inputs=Textbox(lines=2, placeholder="Enter your text here..."),  # text input
        outputs="text",  # text output
        live=False  # model is called only when the submit button is pressed
    )
    # Launch the interface
    interface.launch(share=True)
