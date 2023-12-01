from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from dotenv import dotenv_values
import PyPDF2


config = dotenv_values(".env") 

class User:
    def __init__(self):
        self.chain = None
        self.docsearch = None

users = {}

def upload_and_embed_documents(user_id, file):
    reader = PyPDF2.PdfReader(file)
    number_of_pages = len(reader.pages)
    text = ''
    for page_num in range(number_of_pages):
        page = reader.pages[page_num]
        text += page.extract_text()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=config['OPENAI_API_KEYS'])
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": f"pg-{i}"} for i in range(len(texts))])

    chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff", retriever=docsearch.as_retriever())

    users[user_id].docsearch = docsearch
    users[user_id].chain =chain
    return "Successful"

def query_chain(user_id, question):
    if user_id not in users or users[user_id].chain is None:
        return "User not found or chain not initialized. Please upload documents first."

    result = users[user_id].chain({"question": question}, return_only_outputs=True)
    return result

# API endpoint to create a new user
def create_user_api(user_id):
    if user_id not in users:
        users[user_id] = User()
        return "User created successfully."
    else:
        return "User already exists."

# API endpoint to upload documents and perform embeddings for a user
def upload_documents_api(user_id, file_path):
    if user_id not in users:
        return "User not found. Please create a user first."

    upload_and_embed_documents(user_id, file_path)
    return "Documents uploaded and embeddings created successfully."

# API endpoint to query the chain for a user
def query_chain_api(user_id, question):
    if user_id not in users:
        return "User not found. Please create a user first."

    result = query_chain(user_id, question)
    return result