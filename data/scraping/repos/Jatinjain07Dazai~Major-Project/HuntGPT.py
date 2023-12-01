from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import time

app = Flask(__name__)
load_dotenv()
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

from constants import *

def create_qa_instance():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    callbacks = [StreamingStdOutCallbackHandler()]
    
    if model_type == "LlamaCpp":
        llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    elif model_type == "GPT4All":
        llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    else:
        raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

qa_instance = create_qa_instance()
context = ""  # Initialize an empty context

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    global context
    query = request.form['question']
    if query.strip() == "":
        return render_template('index.html', result="Please enter a question.")

    # Add the context to the query
    input_text = context + "\n" + query

    # Get the answer from the chain
    start = time.time()
    res = qa_instance(input_text)
    answer, docs = res['result'], res['source_documents']
    end = time.time()

    # Update the context with the current interaction (both question and answer)
    context += "\n" + query + "\n" + answer + "\n"

    return render_template('index.html', question=query, answer=answer, docs=docs, time_taken=str(round(end - start, 2)))

if __name__ == '__main__':
    app.run()