import os
import argparse
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run a query on the Vector Store.")
parser.add_argument("query", help="The query to run.")
args = parser.parse_args()

# Load environmental variables
load_dotenv() 

# Set OpenAI and ActiveLoop API keys
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['ACTIVELOOP_TOKEN'] = os.getenv('ACTIVELOOP_TOKEN')

# Specify the model to use
EMBEDDING_MODEL = os.getenv('EMBEDDINGS_MODEL')
ACTIVELOOP_DATASET_PATH = os.getenv('DATASET_PATH')

QA_PROMPT = "As a digital assistant with access to a vast database of dental knowledge, I'm asked: {question}"
CONDENSE_PROMPT = "I have found a variety of resources. Can you summarize the key points about: {documents}"



# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, disallowed_special=())

# Load DeepLake Vector Store
db = DeepLake(dataset_path=ACTIVELOOP_DATASET_PATH, read_only=True, embedding_function=embeddings)

# Create retriever object and specify search parameters
retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 3

# Create a LangChain model for QA
model = ChatOpenAI(
    model='gpt-3.5-turbo', 
    temperature=0.4, 
    max_tokens=2500
)

qa = RetrievalQA.from_llm(
    model, 
    retriever=retriever,
    qa_template=QA_PROMPT, 
    question_generator_template=CONDENSE_PROMPT, 
    return_source_documents=True
)



# Perform a query
response = qa.run(args.query)
print(response)
