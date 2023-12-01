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
# print(st.session_state["OPENAI_API_KEY"])
# parser = argparse.ArgumentParser()
#
# parser.add_argument("--apikey_filepath",default='openapi_key.txt',type=str,help="This is where the api_key is stored as .txt file.")
# parser.add_argument("--documents_path",default="../simpossum/",type=str,help="This is where the pdf documents are stored.")
# args = parser.parse_args()
# os.environ["OPENAI_API_KEY"] = load_api_key(filepath=args.apikey_filepath)

# embeddings



# Layout
st.write("PDF Ninja App")
st.header("📖Here to knock your pdfs off")
uploaded_file = st.file_uploader("Pick a pdf file (⚠️ Currently supports less than 5 pages)",type=["pdf"])



# if not query:
#     query = random.choice(["get me the invoice for garden gnomes","get me Simran's CV"])
#     print("\nWe chose the prompt: "+ query)
# docs = retriever.get_relevant_documents(query)
#
# pp = pprint.PrettyPrinter()
# pp.pprint("".join(docs[0].page_content[:1000].replace("\n"," ")))

query = st.text_area("Ask your questions here. For example, ask ""Tell me more about this file."" ",
                             on_change=clear_submit)

with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file))

    if submitted:
        embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.get("OPENAI_API_KEY"))
        llm = OpenAI(openai_api_key=st.session_state.get("OPENAI_API_KEY"), model_name="text-davinci-003")
        chain = load_qa_chain(OpenAI(openai_api_key=st.session_state.get("OPENAI_API_KEY")), chain_type='stuff')


        docsall = process_file_st(uploaded_file)
        vector_store = FAISS.from_documents(docsall, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 1})  # get top k docs # this can be an argaparser requirement

        # if not query:
        #     query="Why is Simran amazing?"
        with st.spinner("Indexing document... This may take a while⏳"):
            docs_focus = vector_store.similarity_search(query) #we can use the entire docs base but I am focussing the QA on the document in question
            #print(docs_focus)
            st.markdown(chain.run(input_documents = docsall,question=query))




st.stop()


