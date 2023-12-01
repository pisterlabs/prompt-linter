from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load the PDF document using PyPDFLoader
loader = PyPDFLoader("/Users/jorisdejong/Documents/GitHub/ai_unchained/2023_09_16/files/TSLA-Q1-2023-Update.pdf")

# Extract the text data from the PDF document
data = loader.load()

# Split the text into chunks using TokenTextSplitter
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

# Generate embeddings using OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Create a retriever using Chroma and the generated embeddings
retriever = Chroma.from_documents(chunks, embeddings).as_retriever()

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.1)

# Create a RetrievalQA instance with the ChatOpenAI model and the retriever
qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

# While loop to keep asking questions
while True:
    question = input("What is your question? ")
    answer = qa.run(question)
    print(answer)
    print("------\n")
