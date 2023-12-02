from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.output_parsers import PydanticOutputParser

from typing import List
from pydantic import BaseModel, Field
import streamlit as st


import os


openai_key = st.secrets["OPENAI_API_KEY"]
active_loop_key = st.secrets["ACTIVELOOP_TOKEN"]


embeddings = OpenAIEmbeddings()

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613", streaming=True)


# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


def get_multi_query_retriever(file, language='fr'):

    if file["store_directory"][-1] != "/":
        file["store_directory"] += "/"

    if not os.path.exists(file["store_directory"] + file["name"]):
        if file['path'].endswith(".pdf"):
            loader = PyPDFLoader(file["path"])
            pages = loader.load_and_split()
        elif file['path'].endswith('.txt'):
            loader = TextLoader(file["path"])
            pages = loader.load_and_split()
        else:
            raise ValueError("The file extension must be either .pdf or .txt")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=file['chunk_size'],
            chunk_overlap=file['chunk_overlap'],
            length_function=len,
        )

        docs = text_splitter.split_documents(pages)

        retriever = FAISS.from_documents(docs, embeddings)
        retriever.save_local(file["store_directory"] + file["name"])
        retriever = retriever.as_retriever()
    else:
        print(f"The retriever already exists. Loading {file['name']}...")
        retriever = FAISS.load_local(file["store_directory"] + file["name"], embeddings)
        retriever = retriever.as_retriever(search_kwargs={"k": 2})

    # Supplying your own prompt
    # You can also supply a prompt along with an output parser to split the results into a list of queries.

    output_parser = LineListOutputParser()
    if language == "en":
        template = """You are an AI language model assistant. Your task is to generate five 
                different versions of the given user question to retrieve relevant documents from a vector 
                database. By generating multiple perspectives on the user question, your goal is to help
                the user overcome some of the limitations of the distance-based similarity search. 
                Provide these alternative questions separated by newlines.
                Original question: {question}"""

        query_prompt = PromptTemplate(
            input_variables=["question"],
            template=template,
        )
    elif language == "fr":
        template = """Vous êtes un assistant de modèle linguistique IA. Votre tâche consiste à générer cinq différentes 
        versions de la question de l'utilisateur en lien avec un document juridique, par exemple les conditions 
        générales d'une banque. Ces versions doivent permettre de récupérer des documents pertinents d'une base de 
        données vectorielle. En générant plusieurs perspectives sur la question de l'utilisateur, votre objectif est 
        d'aider l'utilisateur à surmonter certaines des limites de la recherche de similarité basée sur la distance.
        Fournissez ces questions alternatives séparées par des sauts de ligne, en prenant en compte la nature légale du 
        document en question.
        Question originale : {question}"""

        query_prompt = PromptTemplate(
            input_variables=["question"],
            template=template,
        )
    else:
        raise ValueError("Language must be either 'en' or 'fr'")

    # Chain
    llm_chain = LLMChain(llm=llm, prompt=query_prompt, output_parser=output_parser)

    # Run
    retriever = MultiQueryRetriever(
        retriever=retriever, llm_chain=llm_chain, parser_key="lines"
    )  # "lines" is the key (attribute name) of the parsed output

    return retriever


def get_multi_query_retriever_deep_lake(file, language='fr'):

    if file["store_directory"][-1] != "/":
        file["store_directory"] += "/"

    if not os.path.exists(file["store_directory"] + file["name"]):
        if file['path'].endswith(".pdf"):
            loader = PyPDFLoader(file["path"])
            pages = loader.load_and_split()
        elif file['path'].endswith('.txt'):
            loader = TextLoader(file["path"])
            pages = loader.load_and_split()
        else:
            raise ValueError("The file extension must be either .pdf or .txt")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=file['chunk_size'],
            chunk_overlap=file['chunk_overlap'],
            length_function=len,
        )

        docs = text_splitter.split_documents(pages)
        # if file['store_directory'] does not exist, it will be created
        if not os.path.exists(file["store_directory"]):
            os.makedirs(file["store_directory"])

        # creating the database
        print("Creating the database this may take a while...")
        db = DeepLake(
            dataset_path=file["store_directory"] + file["name"], embedding_function=embeddings, overwrite=True
        )
        db.add_documents(docs)
        retriever = db.as_retriever()

    else:
        # print(f"The retriever already exists. Loading {file['name']}...")
        db = DeepLake(
            dataset_path=file["store_directory"] + file["name"], embedding_function=embeddings, read_only=True
        )
        retriever = db.as_retriever()

    output_parser = LineListOutputParser()
    if language == "en":
        template = """You are an AI language model assistant. Your task is to generate five 
                different versions of the given user question to retrieve relevant documents from a vector 
                database. By generating multiple perspectives on the user question, your goal is to help
                the user overcome some of the limitations of the distance-based similarity search. 
                Provide these alternative questions separated by newlines.
                Original question: {question}"""

        query_prompt = PromptTemplate(
            input_variables=["question"],
            template=template,
        )
    elif language == "fr":
        template = """Vous êtes un assistant de modèle linguistique IA. Votre tâche consiste à générer cinq différentes 
        versions de la question de l'utilisateur en lien avec un document juridique, par exemple les conditions 
        générales d'une banque. Ces versions doivent permettre de récupérer des documents pertinents d'une base de 
        données vectorielle. En générant plusieurs perspectives sur la question de l'utilisateur, votre objectif est 
        d'aider l'utilisateur à surmonter certaines des limites de la recherche de similarité basée sur la distance.
        Fournissez ces questions alternatives séparées par des sauts de ligne, en prenant en compte la nature légale du 
        document en question.
        Question originale : {question}"""

        query_prompt = PromptTemplate(
            input_variables=["question"],
            template=template,
        )
    else:
        raise ValueError("Language must be either 'en' or 'fr'")

    # Chain
    llm_chain = LLMChain(llm=llm, prompt=query_prompt, output_parser=output_parser)

    # Run
    retriever = MultiQueryRetriever(
        retriever=retriever, llm_chain=llm_chain, parser_key="lines"
    )  # "lines" is the key (attribute name) of the parsed output

    return retriever


def get_multi_query_retriever_deep_lake_cloud(file, language='fr', create_db=False):
    username = "mmoshek"  # your username on app.activeloop.ai
    dataset_path = f"hub://{username}/{file['name']}"  # path to your dataset on app.activeloop.ai

    if create_db:
        if file['path'].endswith(".pdf"):
            loader = PyPDFLoader(file["path"])
            pages = loader.load_and_split()
        elif file['path'].endswith('.txt'):
            loader = TextLoader(file["path"])
            pages = loader.load_and_split()
        else:
            raise ValueError("The file extension must be either .pdf or .txt")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=file['chunk_size'],
            chunk_overlap=file['chunk_overlap'],
            length_function=len,
        )

        docs = text_splitter.split_documents(pages)

        # creating the database
        print("Creating the database this may take a while...")
        db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings, overwrite=True)
        db.add_documents(docs)
        retriever = db.as_retriever()
    else:
        db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings, read_only=True)
        retriever = db.as_retriever()

    output_parser = LineListOutputParser()
    if language == "en":
        template = """You are an AI language model assistant. Your task is to generate five 
                different versions of the given user question to retrieve relevant documents from a vector 
                database. By generating multiple perspectives on the user question, your goal is to help
                the user overcome some of the limitations of the distance-based similarity search. 
                Provide these alternative questions separated by newlines.
                Original question: {question}"""

        query_prompt = PromptTemplate(
            input_variables=["question"],
            template=template,
        )
    elif language == "fr":
        template = """Vous êtes un assistant de modèle linguistique IA. Votre tâche consiste à générer cinq différentes 
        versions de la question de l'utilisateur en lien avec un document juridique, par exemple les conditions 
        générales d'une banque. Ces versions doivent permettre de récupérer des documents pertinents d'une base de 
        données vectorielle. En générant plusieurs perspectives sur la question de l'utilisateur, votre objectif est 
        d'aider l'utilisateur à surmonter certaines des limites de la recherche de similarité basée sur la distance.
        Fournissez ces questions alternatives séparées par des sauts de ligne, en prenant en compte la nature légale du 
        document en question.
        Question originale : {question}"""

        query_prompt = PromptTemplate(
            input_variables=["question"],
            template=template,
        )
    else:
        raise ValueError("Language must be either 'en' or 'fr'")

    # Chain
    llm_chain = LLMChain(llm=llm, prompt=query_prompt, output_parser=output_parser)

    # Run
    retriever = MultiQueryRetriever(
        retriever=retriever, llm_chain=llm_chain, parser_key="lines"
    )  # "lines" is the key (attribute name) of the parsed output

    return retriever