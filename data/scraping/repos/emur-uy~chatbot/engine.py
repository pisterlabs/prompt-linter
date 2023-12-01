import dotenv
import os
import psycopg2
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

# Load environment variables from .env file, including the OpenAI API key
dotenv.load_dotenv()

# Establish a connection to the database using the connection string stored in an environment variable
db_url = os.getenv('SQLALCHEMY_DATABASE_URL')
connection = psycopg2.connect(db_url)
cursor = connection.cursor()

# Execute a SQL query to fetch all data from the 'data' column of the 'information' table
cursor.execute("SELECT data FROM information")
rows = cursor.fetchall()

# Close the database connection when done
connection.close()

# Extract the 'data' field from each row and store them in a list
documents = [row[0] for row in rows]

# Instantiate a CharacterTextSplitter with a specific chunk size and overlap
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Use the text splitter to convert each document into a list of chunks
# Each chunk will be a separate document
docs = text_splitter.create_documents(documents)

# Use the FAISS (Facebook AI Similarity Search) library to create an index from the documents
# This index will allow us to perform fast similarity searches
# Using the OpenAIEmbeddings to transform the text documents into numerical vectors
faissIndex = FAISS.from_documents(docs, OpenAIEmbeddings())

# Save the index to a local file for future use
faissIndex.save_local("db_docs")