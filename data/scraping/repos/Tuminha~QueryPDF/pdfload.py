import os
import glob
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load variables from .env file
load_dotenv()

# Set the ACTIVELOOP_TOKEN, OPENAI_API_KEY, EMBEDDINGS_MODEL, and DATASET_PATH environment variables
ACTIVELOOP_TOKEN = os.getenv('ACTIVELOOP_TOKEN')
EMBEDDINGS_MODEL = os.getenv('EMBEDDINGS_MODEL')
DATASET_PATH = os.getenv('DATASET_PATH')
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')

os.environ['ACTIVELOOP_TOKEN'] = ACTIVELOOP_TOKEN
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Create the DeepLake Vector Store
embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)

# Iterate over all PDF files in the specified directory
pdf_directory = 'docs'
pdf_files = glob.glob(pdf_directory + '/*.pdf')

for pdf_file in pdf_files:
    # Load and split each PDF file
    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split()

    # Add the documents to the Vector Store
    deeplake_db = DeepLake.from_documents(pages, embeddings, dataset_path=DATASET_PATH)

# Create a retriever
retriever = deeplake_db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 2

# Create a RetrievalQA chain and run it
model = ChatOpenAI(model='gpt-3.5-turbo')
qa = RetrievalQA.from_llm(model, retriever=retriever)

# Run the question
response = qa.run('What is the platform switching?')
print(response)

