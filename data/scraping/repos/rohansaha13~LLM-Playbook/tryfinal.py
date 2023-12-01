import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from PIL import Image
import pytesseract
from transformers import pipeline
from langchain.document_loaders import UnstructuredWordDocumentLoader
#import docx2txt
from langchain.document_loaders import DirectoryLoader




st.set_page_config(page_title='Doc-Q&A ',page_icon="./logo.png",layout ="wide") 
image = "./logo.png"
st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

st.header("Chat with Documents 💬")

from streamlit_extras.app_logo import add_logo
add_logo("logo.png", height=60)

import streamlit as st
import os

def save_uploadedfile(uploadedfile):
    with open(os.path.join("Data", uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File: {} to Data".format(uploadedfile.name))

uploadedfiles = st.file_uploader("Upload your Document(pdf,text,docx,images)",accept_multiple_files=True)

for document in uploadedfiles:
    if uploadedfiles is not None:
        save_uploadedfile(document)

pdf_loader = DirectoryLoader('./Data', glob="**/*.pdf")
docx_loader = DirectoryLoader('./Data', glob="**/*.docx")
txt_loader = DirectoryLoader('./Data', glob="**/*.txt")
img_loader = DirectoryLoader('./Data', glob="**/*.png")

model_path = st.selectbox('Select an Option',["google/flan-t5-base","MBZUAI/LaMini-Flan-T5-783M","google/flan-t5-xxl","impira/layoutlm-document-qa",'bigscience/bloom-560m',"google/flan-t5-xl"])

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Anubhav194010\Tesseract-OCR\tesseract.exe"

#take all the loader
loaders = [pdf_loader, docx_loader, txt_loader,img_loader]

#lets create document 
documents = []
for loader in loaders:
    documents.extend(loader.load())

st.write(f'You have {len(documents)} document(s) in your data')
st.write(f'There are {len(documents[0].page_content)} characters in your document')


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=40) #chunk overlap seems to work better
documents = text_splitter.split_documents(documents)
st.write(len(documents))

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
# load the vectorstore
db = FAISS.from_documents(documents, embeddings)
st.write("Embeddings Done")

if model_path.endswith("-qa"):
    nlp = pipeline("document-question-answering",model=model_path)
    query = st.text_input("Ask questions from your documents:")
    # result = nlp(img,query)
    # st.write(result)
        
else:
    #st.write(type(model_path))
    llm = HuggingFaceHub(repo_id=model_path, model_kwargs={"temperature":0.1})
    st.write("Model Loaded - ",model_path)
    # Accept user questions/query
    query = st.text_input("Ask questions from your documents:")
    if query:
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 3}), return_source_documents=False, verbose=False)
        result = qa({"query": query})
        st.write(result)


# #for document in uploaded_files:
#       #new_list = []
#       #st.write(document)
# if document is not None:
#         #st.write(type(document))
#         if document.name.endswith('.pdf'):
#             data = PdfReader(document)
#             text = ""
#             for page in data.pages:
#                   text += page.extract_text()
#               #new_list.append(text)
#               #st.write(chunks)
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap = 10)
#             chunks = text_splitter.create_documents(text)
#             st.write("PDF Loaded")
#         elif document.name.endswith('.txt'):
#               #st.write(type(document))
#             s = document.getvalue().decode()
#             with open('abc.txt', 'x') as f:
#                   f.write(s)
#             text_reader = TextLoader('abc.txt')
#             data = text_reader.load()
#             os.remove("abc.txt")
#               # Text Splitter
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap = 10)
#             chunks = text_splitter.split_documents(data)
#               #st.write(chunks)
#             st.write("Text Loaded")
#         elif document.name.endswith('.png') or document.name.endswith('.jpg') or document.name.endswith('.jpeg'):
#             #   s = document.getvalue().decode()
#             pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Anubhav194010\Tesseract-OCR\tesseract.exe"
#             img = Image.open(document)
#             st.image(img)
#             loader = UnstructuredImageLoader(img)
#             data = loader.load()
#             #st.write(data)
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap = 10)
#             chunks = text_splitter.split_documents(data)

#         elif document.name.endswith('.doc') or document.name.endswith('.docx'):
#             raw_text = docx2txt.process(document)
#             with open('abc.txt', 'x') as f:
#                   f.write(raw_text)
#             text_reader = TextLoader('abc.txt')
#             data = text_reader.load()
#             os.remove("abc.txt")
#               # Text Splitter
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap = 10)
#             chunks = text_splitter.split_documents(data)
              
#       # # embeddings
#         embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#       # load the vectorstore
#         db = FAISS.from_documents(chunks, embeddings)
#         st.write("Embeddings Done")



#         if model_path.endswith("-qa"):
#              nlp = pipeline("document-question-answering",model=model_path)
#              query = st.text_input("Ask questions from your documents:")
#              result = nlp(img,query)
#              st.write(result)
              
#         else:
#          #st.write(type(model_path))
#          llm = HuggingFaceHub(repo_id=model_path, model_kwargs={"temperature":0.1})
#          st.write("Model Loaded - ",model_path)
#          # Accept user questions/query
#          query = st.text_input("Ask questions from your documents:")
#          if query:
#             qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 3}), return_source_documents=False, verbose=False)
#             result = qa({"query": query})
#             st.write(result)