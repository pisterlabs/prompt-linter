# %%
import os
import weaviate
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Weaviate
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
cohere_api_key = os.getenv('COHERE_API_KEY')
weaviate_api_key = os.getenv('weaviate_api_key')
weaviate_url = os.getenv('weaviate_url')


# Connect to the Weaviate demo databse containing 10M wikipedia vectors
# This uses a public READ-ONLY Weaviate API key
auth_config = weaviate.auth.AuthApiKey(api_key=weaviate_api_key) 

client = weaviate.Client( url=weaviate_url, auth_client_secret=auth_config, 
                         additional_headers={ "X-Cohere-Api-Key": cohere_api_key})


vectorstore = Weaviate(client,  index_name="Articles", text_key="text")
vectorstore._query_attrs = ["text", "title", "url", "views", "lang", "_additional {distance}"]
vectorstore.embedding =CohereEmbeddings(model="embed-multilingual-v2.0", cohere_api_key=cohere_api_key)
llm =OpenAI(temperature=0, openai_api_key=openai_api_key)
qa = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
query = "Why is the theory of everything significant?"
result = qa({"query": query})
result['result']

# %%
result['result']

# %%
query = "Why is the theory of everything significant?"
docs = vectorstore.similarity_search(query, 10)
docs

# %%
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
docs1 = retriever.get_relevant_documents(query)
docs1

# %%
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
compressor = CohereRerank(model='rerank-multilingual-v2.0', top_n=4 )
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
compressed_docs = compression_retriever.get_relevant_documents("Why is the theory of everything significant?")
compressed_docs

# %%
qa = RetrievalQA.from_chain_type(llm, retriever=compression_retriever)
result = qa({"query": query})

# %%
result['result']

# %%
from langchain.prompts import PromptTemplate

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer in {language}:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question","language"])

# %%
qa = RetrievalQA.from_chain_type(llm, retriever=compression_retriever, chain_type_kwargs={"prompt": PROMPT.partial(language="english")})
result = qa({"query": query})
result['result']


