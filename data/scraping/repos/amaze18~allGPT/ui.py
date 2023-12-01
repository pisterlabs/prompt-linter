import gradio as gr
import random
import time

messages = []
kill = 0

#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse

from time import sleep
from multiprocessing import Process



load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')

from constants import CHROMA_SETTINGS

def chat_gpt(question):
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {model_type} not supported!")
            exit;
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    # Interactive questions and answers
    
    query = question #input("\nEnter a query: ")

    # Get the answer from the chain
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
    chatbot = gr.Chatbot(label="Chat with your Professor")      # displays a chatbot
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
        #process = Process(target=chat_gpt(message))
        # run the process
        #response =process.start()
        # if(kill == 1):
        #     print("killing::", kill)
        #     kill = 0
        #     # terminate the process
        #     process.terminate()
        #     print("killed::", kill)
        
        response, doc, metadata = chat_gpt(message)

        
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