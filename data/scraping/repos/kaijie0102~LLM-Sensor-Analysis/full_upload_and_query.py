"""
This script is used to load a pdf file in vector form into pinecone.
For further querying, use the openai_pinecone.py script
"""

from dotenv import load_dotenv,find_dotenv
from langchain.llms import OpenAI
import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import pinecone
import json
import PyPDF2

# load environment variables
load_dotenv(find_dotenv())
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
PINECONE_ENV=os.environ["PINECONE_ENV"]
PINECONE_API_KEY=os.environ["PINECONE_API_KEY"]

# load pdf file and split it into chunks
loader = UnstructuredFileLoader("data/vibration_prompt_steven_pdf.pdf")
data = loader.load()
# print("data: ",data)

# Load prompt file
# filename1 = 'light_prompt_try.txt'
# with open(filename1, 'r') as f:
#     data = f.read()
        # print("length of prompt!", len(prompt1))  # length is like 9thousand
        # messages.append({"role": "user", "content": "{}".format(prompt1)})

# query_vector = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(data)
# texts = text_splitter.split_text(data)

# print("TEXTS: ",texts)
# OpenAI’s text embedding engine to embed data into vectors
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

# go to pinecone and create index. Choose [dimension=1536, Metric=cosine, Pod Type = P1
# Enter name of index in the string below
index_name = "langchain1"

# get embeddings and then pass to pinecone
docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
print("Successfully upserted {} vectors".format(len(texts)))
# docsearch = Pinecone.from_texts([t for t in texts], embeddings, index_name=index_name)
# docsearch <langchain.vectorstores.pinecone.Pinecone object at 0xaab55ab0>
# print("DOCSEARCH: ", json.dumps(docsearch))

llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")

# # Load mqtt file
filename3 = 'mqtt_message.txt'
with open(filename3, 'r') as f:
    prompt3 = f.read()
    # print("PROMPT3: ", prompt3)

query = prompt3
docs =docsearch.similarity_search(query,k=3)
# docs = docsearch.get_relevant_documents(query)

# print("DOCS: ",type(docs))
# print("QUERY: ",type(query))

print("Prompt has been sent. Waiting for response...")
response = chain.run(input_documents=docs, question=query)
print(response)