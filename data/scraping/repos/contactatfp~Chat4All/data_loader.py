import json
import os
import pinecone
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredEmailLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders import GoogleDriveLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.prompts import load_prompt
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from flask import session
import forms
from langchain.document_loaders import YoutubeLoader




with open('config.json') as f:
    config = json.load(f)

with open('static/persona/tutor.json') as p:
    tutor = json.load(p)

with open('static/persona/business.json') as p:
    business = json.load(p)

openai.api_key = config['openai_api_key']
pinecone_api_key = config['pinecone_api_key']
pinecone_environment = config['pinecone_environment']


# initialize pinecone
pinecone.init(
    api_key=pinecone_api_key,  # find at app.pinecone.io
    environment=pinecone_environment,  # find at app.pinecone.io
)






def load_microsoft_word(path):
    loader = Docx2txtLoader(path)
    pages = loader.load_and_split()
    embed_new_docs(pages)
    return pages


def load_images(path):
    loader = UnstructuredImageLoader(path, mode="elements")
    pages = loader.load()
    # print(pages[0])
    embed_new_docs(pages)
    return pages


def excel_loader(path):
    loader = UnstructuredExcelLoader(path, mode="elements")
    pages = loader.load_and_split()
    embed_new_docs(pages)
    return pages


def load_youtube(url):
    loader = YoutubeLoader.from_youtube_url(
        youtube_url=url,
        add_video_info=True,
        language=["en", "id"],
        translation="en",
    )
    pages = loader.load_and_split()
    embed_new_docs(pages)
    return pages


def email_loader(path):
    loader = UnstructuredEmailLoader(path)
    pages = loader.load_and_split()
    embed_new_docs(pages)
    return pages


def csv_loader(path):
    loader = CSVLoader(file_path=path)
    pages = loader.load_and_split()
    embed_new_docs(pages)
    return pages


def google_drive_loader(folder_id):
    loader = GoogleDriveLoader(
        folder_id=folder_id,
        # Optional: configure whether to recursively fetch files from subfolders. Defaults to False.
        recursive=False,
    )
    pages = loader.load_and_split()
    embed_new_docs(pages)
    return pages


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    embed_new_docs(pages)
    return pages


def load_txt(file_path):
    loader = TextLoader(file_path)
    loader.load()
    pages = loader.load_and_split()
    embed_new_docs(pages)
    return pages



# Import other loaders as needed

def load_document(file_path):
    # Determine the file extension
    file_extension = file_path.split('.')[-1].lower()
    # Call the appropriate loader based on the file extension
    if file_extension == 'pdf':
        loader = load_pdf(file_path)
    elif file_extension == 'docx':
        loader = load_microsoft_word(file_path)
    elif file_extension == 'jpg' or file_extension == 'jpeg' or file_extension == 'png':
        loader = load_images(file_path)
    elif file_extension == 'csv':
        loader = csv_loader(file_path)
    elif file_extension == 'doc':
        loader = load_microsoft_word(file_path)
    elif file_extension == 'xlsx':
        loader = excel_loader(file_path)
    elif file_extension == 'eml':
        loader = UnstructuredEmailLoader(file_path)
    elif file_extension == 'txt':
        loader = load_txt(file_path)
    elif file_extension[-1:] == '/':
        loader = directory_loader(file_path)
        for i in loader:
            data = embed_new_docs(loader)

        return data

    else:
        raise Exception(f'No loader found for file extension {file_extension}')

    # Load the data
    # data = loader.load_and_split()

    return loader


# for directory
def directory_loader(path):
    text_loader_kwargs = {'autodetect_encoding': True}
    loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    pages = loader.load_and_split()
    embed_new_docs(pages)
    return pages


# for pdfs
# loader = PyPDFLoader("../static/text/Costigan Test.pdf")
# pages = loader.load_and_split()

def split_docs(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    docs = text_splitter.split_documents(pages)
    return docs


def embed_new_docs(pages):
    embeddings = OpenAIEmbeddings(model="ada",openai_api_key=config['openai_api_key'])
    index_name = "chat-all"
    docs = split_docs(pages)
    docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    return docsearch


def embedded_docs():
    embeddings = OpenAIEmbeddings(model="ada", openai_api_key=config['openai_api_key'])
    index_name = "chat-all"
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    return docsearch


# create a new index


# if you already have an index, you can load it like this
# docsearch = Pinecone.from_existing_index(index_name, embeddings)

# from langchain.vectorstores import Chroma
# docsearch = Chroma.from_documents(docs, embeddings)

# def ask_your_model(query, new_docs=None):
#     if new_docs != None:
#         load_document(new_docs)
#     llm = OpenAI(temperature=0.2, input_variables=[])
#     # setup a prompt template from langchain
#     # PROMPT = PromptTemplate(
#     #     input_variables=[],  # update this line
#     #     template="I am kyle, an AI tutor created by OpenAI to help students learn. My role is to quiz you on topics from the documents in the RetrievalQA dataset and explain concepts you may find confusing. Please feel free to ask me questions about anything in the RetrievalQA dataset, and I will do my best to provide helpful explanations and examples.\n\nFor example, you could say:\n\n\"Kyle, can you quiz me on the key events of World War 1?\"\n\"Kyle, I'm having trouble understanding how neural networks work. Can you explain them to me?\"\n\"Kyle, what were the major causes of the Great Depression?\"\nI will respond as an expert tutor would, asking follow up questions, providing explanations, examples and analogies to help strengthen your understanding. My goal is for you to walk away from our conversation with a deeper grasp of the topics we discuss.\n\nTell me when you're ready to begin and I will ask you your first question."
#     # )
#
#     PROMPT = load_prompt("static/persona/tutor.json")
#     PROMPT.format(name="Loudy")
#     # chain_type_kwargs = {"prompt": PROMPT, "document_variable_name": "text"}
#     context = embedded_docs().similarity_search(query)
#     qa = RetrievalQA.from_chain_type(
#         # context=context,
#         llm=llm,
#         chain_type="stuff",
#         retriever=embedded_docs().as_retriever(search_kwargs={"k": 5}),
#         # chain_type_kwargs=chain_type_kwargs
#     )
#     # qa.set_context(persona['tutor'])
#     answer = qa.run(query)
#     return answer


    # return answer

# loader = load_document("../static/text/01 Customer Gets Rid of Your Problem.txt")
#
# query = "How do you get the shields down? Answer that then write me a generic intro to the customer to get their shields down."
# # docs = embedded_docs().similarity_search(query)
#
# # defining LLM
# llm = OpenAI(temperature=0.2)
#
# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
#                                  retriever=embedded_docs().as_retriever(search_kwargs={"k": 2}))
# print(qa.run(query))
