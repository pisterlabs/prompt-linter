import os
import openai
import json

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from flask import Flask, request, make_response
from flask_cors import CORS, cross_origin

API_KEY = os.getenv('AZURE_OPENAI_API_KEY') 
RESOURCE_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')

openai.api_type = 'azure'
openai.api_key = API_KEY if API_KEY else ''
openai.api_base = RESOURCE_ENDPOINT if RESOURCE_ENDPOINT else 'https://<your-api-name>.openai.azure.com'
openai.api_version = '2023-03-15-preview'

os.environ['OPENAI_API_TYPE'] = openai.api_type
os.environ['OPENAI_API_BASE'] = openai.api_base
os.environ['OPENAI_API_KEY'] = openai.api_key
os.environ['OPENAI_API_VERSION'] = openai.api_version

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# ---------------------------------------------------------------------------- #
#                               Parameters to set                              #
# ---------------------------------------------------------------------------- #

# Set the default code base to be used if it not specified in the request
default_code_base = ''
# Update the name of the index for all your code base.
# The key is the code base label and the value is the index name.
index_names = { default_code_base: '' }

# Set the Redis URL to be used if it not specified in the request
index_url = 'redis://<login>:<password>@<host>:6379'

# Set the number of results to be returned by the retriever
retriever_k = 20

# Set the name of the GPT model to be used for the chatbot
gpt_model_name = ''

# Set the name of the embeddings model to be used for the retriever
embeddings_model_name = 'text-embedding-ada-002'

# ---------------------------------------------------------------------------- #
#                              API implementation                              #
# ---------------------------------------------------------------------------- #

embeddings = OpenAIEmbeddings(model=embeddings_model_name, chunk_size=1)

@app.route('/ask', methods=['POST'])
@cross_origin()
def ask():
  if request.method == 'POST':
    data = json.loads(request.data, strict=False)
    
    # Get the code base and its index name to be used
    code_base = data['code_base'] if 'code_base' in data else default_code_base
    code_base_index_name = index_names[code_base] if code_base in index_names else index_names[default_code_base]

    # Create the conversional chain
    redis =  Redis.from_existing_index(embeddings, redis_url=index_url, index_name=code_base_index_name)
    retriever = redis.as_retriever()
    retriever.k = retriever_k

    chat_model = AzureChatOpenAI(deployment_name=gpt_model_name, temperature=0.0, model_kwargs={ 'top_p': 1.0 })
    qa = ConversationalRetrievalChain.from_llm(chat_model, retriever=retriever, return_source_documents=True)

    # Ask the question
    question = data['question']
    _chat_history = data['chat_history']
    chat_history = []
    # chat history needs to be a list of tuples
    for chat in _chat_history:
      chat_history.append((chat[0], chat[1]))
    qa_response = qa({'question': question, 'chat_history': chat_history})

    # Create the response
    sources = qa_response['source_documents']
    response_content = {
      'answer': qa_response['answer'],
      'source_documents': []
    }
    sourceKeys = []
    for source in sources:
      sourceFile = source.metadata['source']
      fileName = source.metadata['fileName']
      if not sourceFile in sourceKeys:
        sourceKeys.append(sourceFile)
        response_content['source_documents'].append({
          'source': sourceFile,
          'fileName': fileName
        })
    response = make_response(response_content, 200)

    return response
  else:
    error = 'Invalid request'
    return make_response(error, 400)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)