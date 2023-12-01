import gradio as gr
import random
import time

from privateGPT import parse_arguments

messages = []
kill = 0

#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma 
from langchain.docstore.document import Document
from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
import os
import glob
import argparse

from time import sleep
from multiprocessing import Process
chunk_size = 256 #512
chunk_overlap = 50
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp


load_dotenv()

#embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
#persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')

from constants import CHROMA_SETTINGS

global qa
args = parse_arguments()
#embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
model_name = "sentence-transformers/all-mpnet-base-v2"
#model_name = "sentence-transformers/LaBSE"
#model_name= 'intfloat/e5-large-v2'
model_name = 'all-MiniLM-L6-v2'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

documents = []

a=glob.glob("source_documents/*.txt")
for i in range(len(a)):
        print(a[i])
        documents.extend(TextLoader(a[i]).load())
        print(TextLoader(a[i]).load())

a=glob.glob("source_documents/*.html")
for i in range(len(a)):

        documents.extend(UnstructuredHTMLLoader(a[i]).load())

a=glob.glob("source_documents/*.pdf")
for i in range(len(a)):

        documents.extend(PDFMinerLoader(a[i]).load())

a=glob.glob("source_documents/*.csv")
for i in range(len(a)):

        documents.extend(CSVLoader(a[i]).load())

a=glob.glob("source_documents/*.ppt")
for i in range(len(a)):

        documents.extend(UnstructuredPowerPointLoader(a[i]).load())

a=glob.glob("source_documents/*.pptx")
for i in range(len(a)):

        documents.extend(UnstructuredPowerPointLoader(a[i]).load())


a=glob.glob("source_documents/*.docx")
for i in range(len(a)):

        documents.extend(UnstructuredWordDocumentLoader(a[i]).load())

a=glob.glob("source_documents/*.ppt")
for i in range(len(a)):

        documents.extend(UnstructuredPowerPointLoader(a[i]).load())

#db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
#retriever = db.as_retriever()
# activate/deactivate the streaming StdOut callback for LLMs
callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
# Prepare the LLM

        

text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
texts = text_splitter.split_documents(documents)
db = Chroma.from_documents(texts, hf)
model_id="sentence-transformers/all-MiniLM-L6-v2"
model_id='digitous/Alpacino30b'
#model_id="google/flan-t5-base"
#model_id='tiiuae/falcon-40b'
model_id="google/flan-t5-large"
llm =  HuggingFacePipeline.from_model_id(model_id=model_id, task="text2text-generation", model_kwargs={"temperature":3e-1, "max_length" : chunk_size}) #, trust_remote_code=True)
#llm =  HuggingFacePipeline.from_model_id(model_id=model_id, task="question-answering", model_kwargs={"temperature":1e-1, "max_length" : 512}) 
retriever = db.as_retriever(search_type='similarity', search_kwargs={"k": 3} )
#do not increase k beyond 3, else 
callbacks = []  
#qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=retriever, return_source_documents=True)
#qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, return_source_documents=True)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)



def chat_gpt(qa, question):
    args = parse_arguments()
    query = question 
    res = qa(query)
    answer, docs = res['result'], [] if args.hide_source else res['source_documents']


    # Print the relevant sources used#ggml-gpt4all-j-v1.3-groovy.bin for the answer
    for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            #print(document.page_content)
            metadata=document.metadata["source"]
            doc=document.page_content
    
    return answer, doc,metadata

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()




with gr.Blocks() as mychatbot:  # Blocks is a low-level API that allows 
                                # you to create custom web applications
    chatbot = gr.Chatbot(label="Chat with Professor Bhagwan: Ask questions to your documents without an internet connection")      # displays a chatbot
    question = gr.Textbox(label="Question")     # for user to ask a question
    clear = gr.Button("Clear Conversation History")  # Clear button
    kill = gr.Button("Stop Current Search")  # Clear button
    
    # function to clear the conversation
    def clear_messages():
        global messages
        messages = []    # reset the messages list


   
    
    def kill_search():
        print("killing")
        kill =1
        #exit()    # reset the messages list
        
    def chat(message, chat_history, kill):
        global messages
        print("chat function launched...")
        print(message)
        messages.append({"role": "user", "content": message})
        response, doc, metadata = chat_gpt(qa, message)

        
        print("private gpt response recieved...")
        print(response)
        content = response + "\n" + "Sources:"+ "\n >" + metadata+ ":" +"\n"  +"Content: "+"\n"  +doc 
        #['choices'][0]['message']['content']
        
        
        chat_history.append((message, content))
        return "", chat_history

    # wire up the event handler for Submit button (when user press Enter)
    question.submit(fn = chat, 
                    inputs = [question, chatbot], 
                    outputs = [question, chatbot])

    # wire up the event handler for the Clear Conversation button

    clear.click(fn = clear_messages, 
                inputs = None, 
                outputs = chatbot, 
                queue = False)
    kill.click(fn = kill_search, 
                inputs = None, 
                outputs = chatbot, 
                queue = False)

mychatbot.launch(share=True)