import os
import pickle
import re

from langchain.docstore.document import Document
from langchain.document_loaders import PagedPDFSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from bot_files.config import setup_key

setup_key()#OpenAI key
ResumeLocation = os.getcwd()
ResumeLocation+='/files/'
re.sub(r"\\\\", "/",ResumeLocation)
    
def get_pdf_data(doc_name): #Function which loads PDF files(Along with page numbers)    
    f_doc_name = ResumeLocation + doc_name
    loader = PagedPDFSplitter(f_doc_name)
    pages = loader.load_and_split()
    return(pages)

def get_resumes():
    sources = []
    for file in os.listdir(ResumeLocation):
        sources.append(get_pdf_data(file))  
    return sources

def create_chunks():
    sources = get_resumes()
    source_chunks = []  #Creating chunks of the source documents    
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
    for source1 in sources: #For each of the source mentioned
        for source in source1: #Go to each page
            for chunk in splitter.split_text(source.page_content): #Chunk up the content in each page
                chunk = chunk + "Filename is " + source.metadata['source']
                source_chunks.append(Document(page_content=chunk, metadata=source.metadata))#Append it to source_chunks """
    return source_chunks

def create_embeddings():
    embeddings = OpenAIEmbeddings()

    source_chunks = create_chunks() 
    
    docsearch = FAISS.from_documents(source_chunks, embeddings)
    docsearch.save_local("faiss_index") 

if __name__ == "__main__":
    create_embeddings()

#Create pickle file for faiss index
""" with open ("search_index.pickle","wb") as f:
    pickle.dump(docsearch,f) """

# #Incomplete: Updating embeddings
""" def update_files():
    # global docsearch 
    with open("search_index.pickle", "rb") as f:
        docsearch = pickle.load(f)
      docsearch.add_documents()
 """
#Load Pickle file
""" with open("search_index.pickle", "rb") as f:
    docsearch = pickle.load(f) """