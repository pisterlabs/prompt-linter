import os

import pinecone
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_API_ENV = os.environ["PINECONE_API_ENV"]

print(f"\n Loading PDF... \n")
loader = UnstructuredPDFLoader("../data/syllabus/CS161_F23.pdf")

# Load the PDF
data = loader.load()

print(f"\n Loaded {len(data)} documents \n")
print(f"\n There are {len(data[0].page_content)} characters in the document \n")

print(f"\n Splitting...\n")

# Chunk the data into smaller pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

print(f"\n Successsfully Split. You now have {len(texts)} documents \n")

# Create embeddings of the documents to get ready for semantic search
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize Pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at https://www.pinecone.io/
    environment=PINECONE_API_ENV,  # find next to the API key
)

index_name = "cs161"

if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, metric="cosine", dimension=1536)

docsearch = Pinecone.from_documents(texts, embeddings, index_name=index_name)

# Get the most similar documents
query = "Who are the instructors for this course?"
docs = docsearch.similarity_search(query)

# Get a natural language answer to the question
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")

llm_response = chain.run(input_documents=docs, question=query)
print(x)
