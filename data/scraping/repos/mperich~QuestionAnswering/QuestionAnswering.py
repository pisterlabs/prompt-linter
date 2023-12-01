import os
import json
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from abc import ABC, abstractmethod
from flask import Flask, request

# Directory containing your PDF files
pdf_directory = os.path.join(os.getcwd(), 'reference_materials')  # replace with your PDF directory path
configFilePath = os.path.join(os.getcwd(), 'config.json')

def load_config(configFilePath):
    try:
        with open(configFilePath) as f:
            config = json.load(f)
    except FileNotFoundError:
        raise ValueError(f'{configFilePath} not found')
    except KeyError as e:
        raise ValueError(f'{configFilePath} is missing the "{e.args[0]}" property')
    
    return config

def get_config_value(config, key):
    try:
        value = config[key]
    except KeyError:
        raise ValueError(f'config.json is missing the "{key}" property')
    
    return value

config = load_config(configFilePath)
apiKey = get_config_value(config, 'apiKey')

def load_chroma_database(pdf_directory, embeddings):
    print("Loading Chroma database...")
    # Load the PDF documents
    loader = PyPDFDirectoryLoader(pdf_directory)
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Load the documents into Chroma
    db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

    # Save the Chroma database to disk
    db.persist()

    return db

class QuestionAnswerer(ABC):
    @abstractmethod
    def answer_question(self, question):
        pass

class SimilaritySearchAnswerer(QuestionAnswerer):
    def __init__(self, db):
        self.db = db

    def answer_question(self, question):
        print("Answering question using similarity search...")
        return self.db.similarity_search(question, k=1)

class QaChainAnswerer(QuestionAnswerer):
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain

    def answer_question(self, question):
        print("Answering question using QA chain...")
        prompt = "Only use data from the current retriever. Only give direct quotes from the current retriever. Do not add any additional context. "
        response = self.qa_chain({"query": prompt + question})
        return response

print("Initializing OpenAI embedding function...")
embeddings = OpenAIEmbeddings(openai_api_key=apiKey)

db = load_chroma_database(pdf_directory, embeddings)

print("Initializing LLM...")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=apiKey)

print("Initializing QA chain...")
qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

print("Initializing question answerers...")
similarity_search_answerer = SimilaritySearchAnswerer(db)
qa_chain_answerer = QaChainAnswerer(qa_chain)

app = Flask(__name__)

# Define the route for answering questions
@app.route('/answer_question', methods=['POST'])
def answer_question():
    question = request.json['question']
    answer_from_qa_chain = qa_chain_answerer.answer_question(question)
    return {
        'answer_from_qa_chain': answer_from_qa_chain
    }

# Run the Flask app
if __name__ == '__main__':
    app.run()