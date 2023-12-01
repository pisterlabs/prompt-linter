import flask
import lancedb
import openai
import langchain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import LanceDB
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os
from flask import Flask, render_template, request, jsonify
from flask import Response

from flask_cors import CORS

app = Flask(__name__)
# cors = CORS(app, resources={r"/*": {"origins": "*"}})

# from dotenv import dotenv_values
# env_vars = dotenv_values('.env')
env_vars = {'OPENAI_API_KEY': 'sk-1234'}


uri = "~/.lancedb"
db = lancedb.connect(uri)
app = Flask(__name__)

OPENAI_API_KEY = env_vars['OPENAI_API_KEY']


# Route for "/" for a web-based interface to this micro-service:
@app.route('/')
def index():
  return "Hello, World"
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')

    return response

@app.route('/test', methods=['POST'])
def test():
    # Process the request
    data = request.get_json()
    print(data)
    print("TESTING")

    # Generate the response
    response_data = {'message': 'Hello, World'}
    response = jsonify(response_data)

    return response

# Handle OPTIONS request for CORS preflight
@app.route('/test', methods=['OPTIONS'])
def handle_options():
    response = jsonify({'message': 'Preflight request received'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    print('Testing')
    return response

@app.route('/store', methods = ['POST'])
def store_embedding():
    json_data = request.get_json()
    text = json_data['raw_text']
    metadata = json_data['metadata']
    document = Document(page_content=text, metadata=metadata)
    chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_documents([document])
    embeddings = OpenAIEmbeddings()

    table = db.create_table("text", data=[
        {"vector": embeddings.embed_query("Hello World"), "text": "Hello World", "id": "1"}
    ], mode="overwrite")

    docsearch = LanceDB.from_documents(chunks, embeddings, connection=table)
    
    return "Loaded Document Into Table"


@app.route('/retrieve', methods = ['POST'])
def retrieve_embedding():
   json_data = request.get_json()
   query = json_data['query']

   assert len(openai.Model.list()["data"]) > 0
   
   def embed_func(query):
        rs = openai.Embedding.create(input=query, engine="text-embedding-ada-002")
        return [record["embedding"] for record in rs["data"]]
   
   query_vector = embed_func([query])[0]
   table = db.open_table("text")
   docs = table.search(query_vector).limit(10).to_df()
   print(docs)
   return docs



if __name__ == '__main__':
    app.run(debug=True)





