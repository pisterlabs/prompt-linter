# Derived from the implementation guide one shot. https://github.com/timescale/vector-cookbook/blob/main/intro_langchain_pgvector/langchain_pgvector_intro.ipynb
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores.pgvector import PGVector, DistanceStrategy
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from IPython.display import Markdown, display
import tiktoken

load_dotenv(find_dotenv())
OPENAI_API_KEY  = os.environ['OPENAI_API_KEY']

host= os.environ['TIMESCALE_HOST']
port= os.environ['TIMESCALE_PORT']
user= os.environ['TIMESCALE_USER']
password= os.environ['TIMESCALE_PASSWORD']
dbname= os.environ['TIMESCALE_DBNAME']

CONNECTION_STRING = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"

df = pd.read_csv('blog_posts_data.csv')

def num_tokens_from_string(string: str, encoding_name = "cl100k_base") -> int:
    if not string:
        return 0
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

text_splitter = TokenTextSplitter(chunk_size=512,chunk_overlap=103)
new_list = []
for i in range(len(df.index)):
    text = df['content'][i]
    token_len = num_tokens_from_string(text)
    if token_len <= 512:
        new_list.append([df['title'][i], df['content'][i], df['url'][i]])
    else:
        split_text = text_splitter.split_text(text)
        new_list.extend(
            [df['title'][i], split_text[j], df['url'][i]]
            for j in range(len(split_text))
        )
df_new = pd.DataFrame(new_list, columns=['title', 'content', 'url'])
df_new.to_csv('blog_posts_data_chunked.csv', index=False)

loader = DataFrameLoader(df_new, page_content_column = 'content')
docs = loader.load()

embeddings = OpenAIEmbeddings()

db = PGVector.from_documents(
    documents= docs,
    embedding = embeddings,
    collection_name= "blog_posts",
    distance_strategy = DistanceStrategy.COSINE,
    connection_string=CONNECTION_STRING)

retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(temperature = 0.0, model = 'gpt-3.5-turbo-16k')

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True,
)

query =  "How does Edeva use continuous aggregates?"
response = qa_stuff.run(query)
display(Markdown(response))

qa_stuff_with_sources = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    verbose=True,
)

responses = qa_stuff_with_sources({"query": query})

def construct_result_with_sources():
    result = responses['result']
    result += "\n\n"
    result += "Sources used:"
    for i in range(len(source_content)):
        result += "\n\n"
        result += source_metadata[i]['title']
        result += "\n\n"
        result += source_metadata[i]['url']
    return result
display(Markdown(construct_result_with_sources()))
