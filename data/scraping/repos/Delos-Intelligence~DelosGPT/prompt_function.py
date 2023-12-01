import streamlit as st
import tempfile

from langchain.document_loaders import UnstructuredFileLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain, RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

import file_function as ffunc

OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']


prompt_template = """Voici des extraits d'un document qu'un utilisateur a renseigné sur la plateforme. 
Utilise ces extraits pour répondre à la question qui t'es posée. 
Fournis des réponses détaillées en les expliquant et en listant le plus d'informations possibles. 
Si les éléments ne suffisent pas ou que tu n'as pas la réponse, dis que tu ne sais pas répondre. N'essaie pas de créer une fausse réponse. 

{context}

Question: {question}
Answer in French:"""


def load_model():
    llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k')
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return llm, embeddings, prompt

def load_retrieval(llm, docsearch, prompt):
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents = True, chain_type_kwargs={"prompt":prompt}, verbose = True)
    return qa

def create_vector_database(pdffile, embed_model):
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(pdffile.read())
        tmpfile_path = tmpfile.name

    pages = ffunc.get_pages_from_pdf(tmpfile_path)
    temp_file_paths = ffunc.create_temp_files_from_strings(pages)
    documents = ffunc.parse_into_pages(temp_file_paths)
    docsearch = FAISS.from_documents(documents, embed_model)
    return docsearch


