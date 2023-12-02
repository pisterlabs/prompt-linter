import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from typing import List
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.document_loaders import WebBaseLoader
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.text_splitter import CharacterTextSplitter


# Web scraping 
def web_query(url_list, query):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-0613",
        temperature=0
    )
    
    # Document loading
    loader = WebBaseLoader(url_list)
    docs = loader.load()
    
    text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
    )
    
    # Document splitting
    docs = text_splitter.split_documents(docs)
    
    # Embedding
    from langchain.embeddings.openai import OpenAIEmbeddings
    embedding = OpenAIEmbeddings()
    
    # Vectorstores
    from langchain.vectorstores import Chroma
    persist_directory = '../vector_db'
    # !rm -rf ./docs/chroma  # remove old database files if any
    vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory=persist_directory
    )
    # docs = vectordb.similarity_search(query,k=3)
    # print(docs[0].page_content)

    # Retrieval
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    # Build prompt
    template = """You are a kind nurse taking care of cancer patient. For answering the patient question you will first determine what is the symptom and severity of the patient then based on context try to educate the patient to address their concerns.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    # Run chain
    qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )   
    result = qa_chain({"query": query})
    print(result["result"])





url_list = [
    "https://www.cancer.gov/about-cancer/treatment/side-effects/pain",
    "https://www.cancer.gov/about-cancer/treatment/side-effects/fatigue",
    "https://www.cancer.gov/about-cancer/treatment/side-effects/flu-symptoms",
    "https://www.cancer.gov/about-cancer/treatment/side-effects/sleep-problems",
]

web_query(url_list, "I am not feeling well today, I am coughing a lot, I also had headache all day long ")