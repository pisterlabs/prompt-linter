import os
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load CSV file
csv_loader = CSVLoader('fishfry-locations.csv')

# Create embeddings
embeddings = OpenAIEmbeddings()

# Create chat model
chat_model = ChatOpenAI()

# Create docstore
docstore = InMemoryDocStore()

# Create index
index = FAISSIndex(embedding_dim=embeddings.embedding_dim)

# Create vector store
vector_store = FAISS(embedding_function=embeddings.embed, index=index, docstore=docstore, index_to_docstore_id={})

# Create conversational retrieval chain
chain = ConversationalRetrievalChain(document_loader=csv_loader, embeddings=embeddings, chat_model=chat_model, vector_store=vector_store)

# Start chat
while True:
    user_input = input("User: ")
    if user_input.lower() == "quit":
        break
    response = chain.get_response(user_input)
    print("Bot: ", response)