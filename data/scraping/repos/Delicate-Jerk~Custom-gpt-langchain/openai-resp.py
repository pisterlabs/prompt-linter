import os

os.environ["OPENAI_API_KEY"] = ""

import langchain

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load and process the text files
# loader = TextLoader('single_text_file.txt')
loader = DirectoryLoader('/content/DB', glob="./*.txt", loader_cls=TextLoader)

documents = loader.load()

# splitting the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

print(len(texts))
print(texts[3])

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

## here we are using OpenAI embeddings but in the future, we will swap out to local embeddings
embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)

# persist the db to disk
vectordb.persist()
vectordb = None

# Now we can load the persisted database from disk and use it as normal.
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)

retriever = vectordb.as_retriever()

retriever = vectordb.as_retriever(search_kwargs={"k": 2})

print(retriever.search_type)
print(retriever.search_kwargs)

# create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)



def calculate_similarity(query, response):
    vectorizer = TfidfVectorizer()
    tfidf_query = vectorizer.fit_transform([query])
    tfidf_response = vectorizer.transform([response])
    similarity = cosine_similarity(tfidf_query, tfidf_response)
    return similarity[0][0]

def process_llm_response(query,llm_response):
    print("answer: ",llm_response['result'])
    # print("Similarity: ", calculate_similarity(query,llm_response['result']))

# Full example
query = ""
while query.lower() != "stop":
    query = input("Enter your question: ")
    llm_response = qa_chain(query)
    process_llm_response(query,llm_response)


