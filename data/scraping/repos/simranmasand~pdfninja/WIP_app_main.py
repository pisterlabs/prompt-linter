from langchain.llms import OpenAI
import streamlit as st
import argparse
import pprint
import random
from tqdm import tqdm
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from utils import *
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate
#----------- load the api key'
api_sidebar()
parser = argparse.ArgumentParser()

parser.add_argument("--apikey_filepath",default='/Users/simranmasand/Downloads/openapi_key.txt',type=str,help="This is where the api_key is stored as .txt file.")
parser.add_argument("--documents_path",default="../simpossum/",type=str,help="This is where the pdf documents are stored.")
args = parser.parse_args()
os.environ["OPENAI_API_KEY"]=load_api_key(filepath=args.apikey_filepath)
st.write("PDF Ninja App")
st.header("📖Here to knock your pdfs off")
file = st.file_uploader("Pick a pdf file",type=["pdf"])

# print(os.environ["OPENAI_API_KEY"])

embeddings = OpenAIEmbeddings()
llm = OpenAI(model_name="text-davinci-003")

# Provide the directory path where you want to search for PDF files

# directory_path = input("Please provide the absolute path of your directory.")




# Call the function to get the list of PDF files in the directory
# pdf_files_list = [file]
# print('-----------------------------------')
# print('These are the files in this folder:')
# print('-----------------------------------')
# # Print the list of PDF files
# for pdf_file in pdf_files_list:
#     print(pdf_file)
#
# print('-----------------------------------')

docsall = process_file_st(file)
vector_store=FAISS.from_documents(docsall,embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 1}) #get top k docs # this can be an argaparser requirement


# if not query:
#     query = random.choice(["get me the invoice for garden gnomes","get me Simran's CV"])
#     print("\nWe chose the prompt: "+ query)
# docs = retriever.get_relevant_documents(query)
#
# pp = pprint.PrettyPrinter()
# pp.pprint("".join(docs[0].page_content[:1000].replace("\n"," ")))

chain = load_qa_chain(OpenAI(),chain_type='stuff')
query = None
end = "END"
while query != end:
    query = st.text_area("What file are you looking for? For example: you can ask get me the invoice for flower bulbs. Or get me Simran's resume. Just press enter for a random prompt ", on_change=clear_submit)
    if not query:
        query="Why is Simran amazing?"
    with st.spinner("Indexing document... This may take a while⏳"):
        docs_focus = vector_store.similarity_search(query) #we can use the entire docs base but I am focussing the QA on the document in question
        #print(docs_focus)
        st.markdown(chain.run(input_documents = docsall,question=query))



st.stop()


