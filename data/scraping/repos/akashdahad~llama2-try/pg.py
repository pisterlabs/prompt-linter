from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import UnstructuredFileLoader
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from langchain.llms import LlamaCpp
from langchain.llms import CTransformers
from langchain import PromptTemplate, LLMChain
import torch
import psycopg2
import pdfplumber
import io


# Database connection parameters
DBNAME = 'postgres'
USER = 'devadmin'
PASSWORD = 'KCE9MP2En93gLCz2'
HOST = 'bhyve-india-dev-db.postgres.database.azure.com'
PORT = '5432'

# Connect to the database
connection = psycopg2.connect(dbname=DBNAME, user=USER, password=PASSWORD, host=HOST, port=PORT)
conn = connection.cursor()

model = SentenceTransformer('bert-base-nli-mean-tokens')

# Extract Text
def extract_data_from_file(url):
  text = ''
  with pdfplumber.open(io.BytesIO(url.content)) as pdf:
    for page in pdf.pages:
      data = page.extract_text(x_tolerance=3, y_tolerance=3)
      text = text + '\n\n\n' + data
  return { "content": text }

# Load Data From Local File
def load_data():
  print("Loading Data")
  loader = UnstructuredFileLoader("./india.txt")
  documents = loader.load()
  return documents

# Split Data into Chunks
def split_data(documents):
  print("Splitting Data")
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
  all_splits = text_splitter.split_documents(documents)
  return all_splits

# Create Embeddings
def create_embeddings(sentences):
  # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
  print("Creating Embeddings")
  embeddings = model.encode(sentences)
  return embeddings

# Store Embeddings
def store_embeddings(docs, embeddings):
  print("Storing Embeddings")
  for doc, embedding in zip(docs, embeddings):
    print(doc)
    st = doc.page_content.replace("'", "") 
    conn.execute(f"INSERT INTO pg_embed (content_id, content,  embedding) VALUES ('{doc.metadata['source']}', '{st}', '{embedding.tolist()}')")
    connection.commit()

# Get Search Results
def get_search_results(query):
  query_embeddings = model.encode([query])[0]
  conn.execute(f"SELECT content FROM pg_embed ORDER BY embedding <-> '{query_embeddings.tolist()}' LIMIT 10")
  results = conn.fetchall()
  for result in results:
    print(result)
  return results

# Load LLM
def load_llm():
    llm = LlamaCpp(
    model_path="./models/llama-2-7b-chat.ggmlv3.q8_0.bin",
    n_gpu_layers=32,
    n_batch=512,
    verbose=True)
    return llm

# Get Answer From LLM
def get_answer_from_llm(content, prompt):
  llm = load_llm()
  print(prompt, content)
  template = PromptTemplate(template=content, input_variables=["question"])
  llm_chain = LLMChain(prompt=template, llm=llm)
  result = llm_chain.run(prompt)
  print(result)
  return result

documents = load_data()

all_splits = split_data(documents)
str_arr = []

for split in all_splits:
  str_arr.append(split.page_content)

embeddings = create_embeddings(str_arr)

print(len(embeddings))
print(embeddings[0])
print(all_splits[0].metadata['source'])

store_embeddings(all_splits, embeddings)

print("Done")