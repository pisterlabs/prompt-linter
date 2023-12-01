from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from flask_cors import CORS
import os
from ingest import main as load_document


load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = int(os.environ.get('MODEL_N_CTX', 1024))
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

from constants import CHROMA_SETTINGS

app = Flask(__name__)
CORS(app)

qa = None
model_type = "GPT4All"

@app.route('/initialize', methods=['POST'])
def initialize_qa():
    global qa
    
    data = request.get_json()
    model_type = data['model_type']
    
    qa = model_init(model_type, model_path, model_n_ctx, model_n_batch)
    
    response = {
        'message': 'QA initialized successfully.'
    }
    
    return jsonify(response)

def model_init(model_type, model_path, model_n_ctx, model_n_batch):
    global qa
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    callbacks = [StreamingStdOutCallbackHandler()]
    
    if model_type == "LlamaCpp":
        llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    elif model_type == "GPT4All":
        llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)

    else:
        raise Exception(f"Model type {model_type} is not supported. Please choose LlamaCpp.")
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    
    return qa


@app.route('/ask', methods=['POST'])
def ask_question():
    global qa, model_type, model_path, model_n_ctx, model_n_batch
    if qa is None:
        model_init(model_type, model_path, model_n_ctx, model_n_batch)
    
    data = request.get_json()
    query = data['query']
    
    # Get the answer from the chain
    res = qa(query)
    answer, docs = res['result'], res['source_documents']
    
    response = {
        'answer': answer,
        'source_documents': [doc.page_content for doc in docs]
    }
    
    return jsonify(response)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        file.save('Backend/source_documents/' + file.filename)
        
        # Call functions to process and load the new document into the document storage mechanism
        load_document()  # Call your custom document loading function
        
        # Assuming you have access to the 'qa' instance, update the retriever with the newly loaded documents
        retriever = qa.llm.retriever
        retriever.update()  # Update the retriever with the new documents
        
        response = {
            'message': 'File uploaded successfully.'
        }
    else:
        response = {
            'error': 'No file provided.'
        }
    return jsonify(response)

if __name__ == '__main__':
    app.run()
