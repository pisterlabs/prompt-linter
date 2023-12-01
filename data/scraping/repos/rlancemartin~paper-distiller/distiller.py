import os
import time
import pypdf
import pickledb
import pandas as pd
import streamlit as st
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain

# Image banner for Streamlit app
st.sidebar.image("Img/sidebar_img.jpeg")

# Get papers (format as pdfs)
papers  = [l.split('.')[0] for l in os.listdir("Papers/") if l.endswith('.pdf')]
selectbox = st.sidebar.radio('Which paper to distill?',papers)
  
# Paper distillation 
class PaperDistiller:

    def __init__(self,paper_name):

        self.name = paper_name
        self.answers = {}
        # Use pickledb as local q-a store (save cost)
        self.cached_answers = pickledb.load('distller.db',auto_dump=False,sig=False) 

    def split_pdf(self,chunk_chars=4000,overlap=50):
        """
        Pre-process PDF into chunks
        Some code from: https://github.com/whitead/paper-qa/blob/main/paperqa/readers.py
        """

        pdfFileObj = open("Papers/%s.pdf"%self.name, "rb")
        pdfReader = pypdf.PdfReader(pdfFileObj)
        splits = []
        split = ""
        for i, page in enumerate(pdfReader.pages):
            split += page.extract_text()
            if len(split) > chunk_chars:
                splits.append(split[:chunk_chars])
                split = split[chunk_chars - overlap:]
        pdfFileObj.close()
        return splits 


    def read_or_create_index(self):
        """
        Read or generate embeddings for pdf
        """

        if os.path.isdir('Index/%s'%self.name):
            print("Index Found!")
            self.ix = FAISS.load_local('Index/%s'%self.name,OpenAIEmbeddings())
        else:
            print("Creating index!")
            self.ix = FAISS.from_texts(self.split_pdf(), OpenAIEmbeddings())
            # Save index to local (save cost)
            self.ix.save_local('Index/%s'%self.name)
            
    def query_and_distill(self,query):
        """
        Query embeddings and pass relevant chunks to LLM for answer
        """

        # Answer already in memory
        if query in self.answers:
            print("Answer found!")
            return self.answers[query]
        # Answer cached (asked in the past) in pickledb
        elif self.cached_answers.get(query+"-%s"%self.name):
            print("Answered in the past!")
            return self.cached_answers.get(query+"-%s"%self.name)
        # Generate the answer 
        else:
            print("Generating answer!")
            query_results = self.ix.similarity_search(query, k=2)
            chain = load_qa_chain(OpenAI(temperature=0.25), chain_type="stuff")
            self.answers[query] = chain.run(input_documents=query_results, question=query)
            self.cached_answers.set(query+"-%s"%self.name,self.answers[query])
            return self.answers[query]
        
    def cache_answers(self):
        """
        Write answers to local pickledb
        """
        self.cached_answers.dump()

# Select paper via radio button 
print(selectbox)
p=PaperDistiller(selectbox)
p.read_or_create_index()

# Pre-set queries for each paper
# TO DO: improve this w/ prompt engineering
queries = ["What is the main innovation or new idea in the paper?",
           "How many tokens or examples are in the training set?",
           "Where is the training set scraped from or obtained and what modalities does it include?",
           "What are the tasks performed by the model?",
           "How is evaluation performed?",
           "What is the model architecture and what prior work used simnilar architecture?"]

# UI headers
headers = ["Innovation","Dataset size","Dataset source","Tasks","Evaluation","Architecture"]

# Outputs
st.header("`Paper Distiller`")
for q,header in zip(queries,headers):
    st.subheader("`%s`"%header)
    st.info("`%s`"%p.query_and_distill(q))
    # time.sleep(3) # may be needed for OpenAI API limit

# Cache the answers 
p.cache_answers()
    