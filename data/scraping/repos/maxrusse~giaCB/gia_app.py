from flask import Flask, render_template, request, jsonify, send_from_directory
import logging
import time
import sys
import os
from llama_index import (GPTVectorStoreIndex, LLMPredictor, 
                         SimpleDirectoryReader, ServiceContext, 
                         StorageContext, load_index_from_storage)
from llama_index import QuestionAnswerPrompt, RefinePrompt
from llama_index.query_engine.transform_query_engine import TransformQueryEngine
from langchain.chat_models import ChatOpenAI
import backoff
import openai
import socket
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor

# Constants
MODEL = "gpt-4"

# Get API key from environment variable or prompt user
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = input("Enter OpenAI API key: ")

# Get folder path from command line argument or prompt user
folder_path = sys.argv[1] if len(sys.argv) > 1 else input("Enter folder path: ")

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = api_key
openai.api_key = api_key

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Initialize LLM Predictor and ServiceContext
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.4, model_name=MODEL))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# Load the index from storage
storage_context = StorageContext.from_defaults(persist_dir=folder_path)  
index = load_index_from_storage(storage_context)

# Define your ENGLISCH_QA_PROMPT and ENGLISCH_REFINE_PROMPT templates here
# ...

# Time function
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return result, elapsed_time
    return wrapper

# Function to handle retries with exponential backoff
@backoff.on_exception(backoff.expo,
                      Exception,
                      max_tries=10,
                      giveup=lambda e: False,  # Don't give up on any exception
                      base=10)
@timeit
def query_index(query_engine, input_text):
    return query_engine.query(input_text)

# Create a QueryEngine
query_engine = index.as_query_engine(
    service_context=service_context,
    similarity_top_k=15,
    response_mode="compact",
    text_qa_template=ENGLISCH_QA_PROMPT,
    refine_template=ENGLISCH_REFINE_PROMPT,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)

def handle_user_query(casedesc):
    input_text = casedesc
    response, response_time_query = query_index(query_engine, input_text)
    output_accGPT = response.response

    filenames_pages = dict()
    for doc_id, metadata in response.metadata.items():
        filename_label = metadata.get("file_name")
        page_label = metadata.get("page_label")
        if filename_label is not None:
            filename_label = os.path.basename(filename_label)
            if filename_label not in filenames_pages:
                filenames_pages[filename_label] = set()
            if page_label is not None:
                filenames_pages[filename_label].add(f"{page_label}")
    summary = []
    for filename, pages in filenames_pages.items():
        sorted_pages = sorted(pages, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))

        summary.append(f"{filename} Page: ({', '.join(sorted_pages)})" if pages else filename)
    return output_accGPT, ", ".join(summary), response_time_query

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    try:
        casedesc = request.form['casedesc']
        result, filenames, response_time_query = handle_user_query(casedesc)
        return jsonify({
            'result': result,
            'filenames': filenames,
            'time': str(response_time_query)
        })
    except Exception as e:
        logging.error("Error while processing the query: ", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/pdfs/<filename>')
def serve_pdf(filename):
    response = send_from_directory(folder_path, filename, as_attachment=False, mimetype='application/pdf')
    response.headers['Content-Disposition'] = f'inline; filename={filename}'
    return response

def find_unused_port(start_port=8000):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                return port
            except OSError:
                port += 1

if __name__ == "__main__":
    port = find_unused_port()
    app.run(host='0.0.0.0', port=port, use_reloader=False)
