from flask import Flask, request, jsonify
import os
import openai
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('.env')

# Initialize Flask app
app = Flask(__name__)

# Set OpenAI API key
#os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

# In-memory storage for past information
memory = {}

# Load documents from a directory
def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

# Split documents into chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

# Load OpenAI embeddings
def load_embeddings():
    client = openai.api_key = os.getenv('OPENAI_API_KEY')
    embeddings = OpenAIEmbeddings(model="ada", client=client)
    return embeddings

# Initialize Pinecone index
def init_index(docs, embeddings):
    pinecone.init(api_key="f3493788-2a36-48ee-a2d3-f6205e2c71c0", environment="asia-northeast1-gcp")
    index_name = "qabot"
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    return index

# Load language model
def load_language_model():
    client = openai.api_key = os.getenv('OPENAI_API_KEY')
    llm = OpenAI(model="gpt-3.5-turbo", client=client)
    return llm

# Initialize conversation history
conversation = []

# Initialize question-answering chain
def initialize_qa_chain(llm):
    chain = load_qa_chain(llm)
    return chain

# Load language model
llm = load_language_model()

# Initialize question-answering chain
chain = initialize_qa_chain(llm)

# Get similar documents based on a query
def get_similar_docs(query, k=1, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs

# Get the answer to a question
def get_answer(query):
    similar_docs = get_similar_docs(query)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer

# Define the API endpoint for answering questions
#@app.route('/answer', methods=['POST'])
# def answer_question():
#     # Get the query from the request data
#     query = request.json['query']
    
#     # Add the query to the conversation history
#     conversation.append(query)
    
#     # Generate response from ChatGPT based on conversation history
#     response = llm.generate(conversation, max_tokens=50)
    
#     # Extract the answer from the response
#     answer = response.choices[0].text.strip()
    
#     # Return the answer as a JSON response
#     return jsonify({'answer': answer})

# Define OpenAI chat model
openai_chat_model = "gpt-3.5-turbo"

# Define the API endpoint for answering questions
@app.route('/answer', methods=['POST'])
def answer_question():
    # Get the query from the request data
    query = request.json['query']

    # Add the query to the conversation history
    conversation.append(query)

    # Generate response from ChatGPT based on conversation history
    response = openai.ChatCompletion.create(
        model=openai_chat_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ],
        max_tokens=50
    )

    # Extract the answer from the response
    answer = response.choices[0].message.content.strip()

    # Return the answer as a JSON response
    return jsonify({'answer': answer})

if __name__ == '__main__':
    # Set the directory path for documents
    directory = 'data'

    # Load documents from the directory
    documents = load_docs(directory)

    # Split documents into chunks
    docs = split_docs(documents)

    # Load OpenAI embeddings
    embeddings = load_embeddings()

    # Initialize Pinecone index
    index = init_index(docs, embeddings)

    # Run the Flask app
    app.run(debug=True)
