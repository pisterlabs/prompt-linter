import os
from typing import List

import openai
from langchain import PromptTemplate

from script.newAzureOpenAI import NewAzureOpenAI

openai.api_type = "azure"
openai.api_base = "https://adt-openai.openai.azure.com/"
openai.api_version = "2022-12-01"
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY", '938ce9d50df942d08399ad736863d063')

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://adt-openai.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "938ce9d50df942d08399ad736863d063"

OPENAI_API_KEY="938ce9d50df942d08399ad736863d063"
PINECONE_API_KEY="33e67396-4ede-4259-b084-73f5cd10098d"
PINECONE_API_ENV="us-east4-gcp"

import os
# os.environ["OPENAI_API_TYPE"] = "azure"
# os.environ["OPENAI_API_BASE"] = "https://<your-endpoint.openai.azure.com/"
# os.environ["OPENAI_API_KEY"] = "your AzureOpenAI key"

# from langchain.embeddings.openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
# text = "This is a test query."
# query_result = embeddings.embed_query(text)

import joblib

from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# loader = UnstructuredPDFLoader("../Foxit PDF SDK Developer Guide.pdf")
# data = loader.load()
# print (f'You have {len(data)} document(s) in your data')
# print (f'There are {len(data[0].page_content)} characters in your document')
#
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(data)

# joblib.dump(texts,'texts.pkl')
# texts = joblib.load('texts.pkl')
#
# print (f'Now you have {len(texts)} documents')



from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

embeddings = OpenAIEmbeddings(document_model_name="text-embedding-ada-002", chunk_size=1)

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV # next to api key in console
)
# index_name = "langchain-openai"
# namespace = "gsdk"

# docsearch = Pinecone.from_texts(
#     [t.page_content for t in texts], embeddings,
#     index_name=index_name, namespace=namespace)
# print("=sss")


docsearch = Pinecone.from_existing_index(index_name, embeddings, namespace=namespace)

from langchain.llms import OpenAI, AzureOpenAI, OpenAIChat
from langchain.chains.question_answering import load_qa_chain

# llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
input_variables=["product"],
template="帮我起一个好听的 {product}名字?",
)

template = """You are a teacher in physics for High School student. Given the text of question, it is your job to write a answer that question with example.
Question: {text}
Answer:
"""
prompt_template = PromptTemplate(input_variables=["text"], template=template)
# answer_chain = LLMChain(llm=llm, prompt=prompt_template)
# answer = answer_chain.run("What is the formula for Gravitational Potential Energy (GPE)?")
# print(answer)




# llm = NewAzureOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
llm = AzureOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

# llm.deployment_name = 'ChatGPT-0301'
llm.deployment_name = 'code-davinci-002'
# llm.max_tokens = 123

from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

answer = chain.run("帮我编写python获取今天天气?")
print(answer)
# chain = load_qa_chain(llm, chain_type="stuff")

# query = "HTML to PDF demo"
# docs = docsearch.similarity_search(query,
#                                    include_metadata=True, namespace=namespace)

# s = chain.run(input_documents=docs, question=query)
# s=chain.run("colorful socks")
# print(s)
