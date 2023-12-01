import os
import openai
import sys
import pinecone
from langchain.chat_models import ChatOpenAI

from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Pinecone

sys.path.append('../..')

_ = load_dotenv(find_dotenv())


def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

openai.api_key = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = ''
PINECONE_ENV = 'us-west4-gcp-free'
index_name = 'image-store-research'
EMBEDDING_MODEL_NAME = 'text-embedding-ada-002'

llm = ChatOpenAI(temperature=0.0)

# embedding model
embed = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)
# connect to index assuming its already created
index = pinecone.Index(index_name)
print('Pinecone index status is', index.describe_index_stats())

text_field = "text"
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

question = 'What are we going to learn from this course?'

# Q&A with the documents using the retrieval QA with chain type = REFINE
from langchain.chains import RetrievalQA, ConversationalRetrievalChain

qa_chain_refine = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type="refine"
)
result = qa_chain_refine({"query": question})
if len(result) > 0:
    print('Result from the RetrievalQA, with chain type = REFINE is ' +  result["result"])