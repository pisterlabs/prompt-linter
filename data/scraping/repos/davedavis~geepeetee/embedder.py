"""Lower level app uses Streamlit to accept a file, split it into chunks,
create embeddings and then use the ChatGPT API to query the doc based on the
embedding search.

Usage: Just run.
"""
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, \
    FAISS

# Initial Setup.
load_dotenv()
chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
chat.model_name = "gpt-4"

# Grab the PDF
reader = PdfReader('./data/2023_GPT4All_Technical_Report.pdf')

# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

# We need to split the text that we read into smaller chunks so that during
# information retrieval we don't hit the token size limits.

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)
print(len(texts))

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)
print(docsearch)

chain = load_qa_chain(OpenAI(), chain_type="stuff")

query = "who are the authors of the article?"
docs = docsearch.similarity_search(query)
print(query)
print(chain.run(input_documents=docs, question=query))

query = "What was the cost of training the GPT4all model?"
docs = docsearch.similarity_search(query)
print(query)
print(chain.run(input_documents=docs, question=query))
