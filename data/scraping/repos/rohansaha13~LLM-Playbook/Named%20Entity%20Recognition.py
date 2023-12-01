import streamlit as st
import spacy
import pandas as pd
from spacy import displacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from flair.data import Sentence
from flair.models import SequenceTagger
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from tabula import read_pdf
from PyPDF2 import PdfReader


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import UnstructuredURLLoader
from langchain import HuggingFaceHub
from langchain.document_loaders import TextLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains.question_answering import load_qa_chain

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

global csv

# Add a section header:
st.set_page_config(page_title="Named Entity Recognition Tagger", page_icon="📘")
st.title("📘 Named Entity Recognition Tagger")
# st.text_input takes a label and default text string:
input_text = st.text_input('Text string to analyze:', 'Jennifer is living in New York and has American Express card.')

new_list=[]
k=[]

# upload a file
document = st.file_uploader("Upload your Document(pdf,text,csv)")

if document is not None:
        #st.write(type(document))
        if document.name.endswith('.pdf'):

            info = PdfReader(document)
            page = info.pages[0]
            s = page.extract_text()
              #df=read_pdf(document)
            st.write("PDF Loaded")

        elif document.name.endswith('.txt'):
              
            s = document.getvalue().decode()
            st.write("Text Loaded")

        elif document.name.endswith('.csv'):


            df=pd.read_csv(document)
            st.write("Csv file loaded")
            st.write(df)

# Function for df convert to csv
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunks = text_splitter.split_text(text)
    return  chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.1, "max_length":64})
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def Roberta_NER(txt):
    global csv
    global df

    
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
    model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")

    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    if document is not None:
        if document.name.endswith('.pdf'):
            st.write("Sentence : ",s)
            df=pd.DataFrame()
            df['Text Input']=pd.Series(s)
            df['Text Input']=df['Text Input'].astype(str)

            if input_ent == "ALL":
                
                df['PERSON']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'PER']).astype(str)
                df['ORG']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'ORG']).astype(str)
                df['LOC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'LOC']).astype(str)
                df['MISC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'MISC']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == "PERSON":
                df['PERSON']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'PER']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == "ORGANISATION":
                df['ORG']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'ORG']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == 'LOCATION':
                df['LOC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'LOC']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == 'MISCELLANEOUS':
                df['MISC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'MISC']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

        elif document.name.endswith('.txt'):
            st.write("Sentence : ",s)
            df=pd.DataFrame()
            df['Text Input']=pd.Series(s)
            df['Text Input']=df['Text Input'].astype(str)

            if input_ent == "ALL":
                
                df['PERSON']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'PER']).astype(str)
                df['ORG']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'ORG']).astype(str)
                df['LOC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'LOC']).astype(str)
                df['MISC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'MISC']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == "PERSON":
                df['PERSON']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'PER']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == "ORGANISATION":
                df['ORG']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'ORG']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == 'LOCATION':
                df['LOC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'LOC']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == 'MISCELLANEOUS':
                df['MISC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'MISC']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)


        elif document.name.endswith('.csv'):
            
            if input_ent == "ALL":
                for i in df.iloc[:,0]:
                    df['PERSON']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'PER']).astype(str)
                    df['ORG']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'ORG']).astype(str)
                    df['LOC']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'LOC']).astype(str)
                    df['MISC']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'MISC']).astype(str)

                   
                st.write(df)
                # Save data
                csv=convert_df(df)


            if input_ent == "PERSON":
                for i in df.iloc[:,0]:
                    df['PERSON']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'PER']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)
                

            if input_ent == "ORGANISATION":
                for i in df.iloc[:,0]:
                    df['ORG']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'ORG']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == "LOCATION":
                for i in df.iloc[:,0]:
                    df['LOC']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'LOC']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)
            
            if input_ent == "MISCELLANEOUS":
                for i in df.iloc[:,0]:
                    df['MISC']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'MISC']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)
    else:
        st.write("Sentence : ",input_text)
        df=pd.DataFrame()
        df['Text Input']=input_text
        df['Text Input']=df['Text Input'].astype(str)

        if input_ent == "ALL":
  
            df['PERSON']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'PER']).astype(str)
            df['ORG']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'ORG']).astype(str)
            df['LOC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'LOC']).astype(str)
            df['MISC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'MISC']).astype(str)
            st.write(df)

        if input_ent == "PERSON":
            df['PERSON']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'PER']).astype(str)
            st.write(df)

        if input_ent == "ORGANISATION":
            df['ORG']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'ORG']).astype(str)
            st.write(df)

        if input_ent == 'LOCATION':
            df['LOC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'LOC']).astype(str)
            st.write(df)

        if input_ent == 'MISCELLANEOUS':
            df['MISC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'MISC']).astype(str)
            st.write(df)

    #Display the entity visualization in the browser:
    #st.markdown(doc, unsafe_allow_html=True)
    
    
def BERT_base(txt):

    global csv
    global df
    
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    if document is not None:
        if document.name.endswith('.pdf'):
            st.write("Sentence : ",s)
            df=pd.DataFrame()
            df['Text Input']=pd.Series(s)
            df['Text Input']=df['Text Input'].astype(str)

            if input_ent == "ALL":
                
                df['PERSON']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'PER']).astype(str)
                df['ORG']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'ORG']).astype(str)
                df['LOC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'LOC']).astype(str)
                df['MISC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'MISC']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == "PERSON":
                df['PERSON']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'PER']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == "ORGANISATION":
                df['ORG']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'ORG']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == 'LOCATION':
                df['LOC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'LOC']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == 'MISCELLANEOUS':
                df['MISC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'MISC']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

        elif document.name.endswith('.txt'):
            st.write("Sentence : ",s)
            df=pd.DataFrame()
            df['Text Input']=pd.Series(s)
            df['Text Input']=df['Text Input'].astype(str)

            if input_ent == "ALL":
                
                df['PERSON']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'PER']).astype(str)
                df['ORG']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'ORG']).astype(str)
                df['LOC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'LOC']).astype(str)
                df['MISC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'MISC']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == "PERSON":
                df['PERSON']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'PER']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == "ORGANISATION":
                df['ORG']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'ORG']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == 'LOCATION':
                df['LOC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'LOC']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == 'MISCELLANEOUS':
                df['MISC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'MISC']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)


        elif document.name.endswith('.csv'):
            
            if input_ent == "ALL":
                for i in df.iloc[:,0]:
                    df['PERSON']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'PER']).astype(str)
                    df['ORG']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'ORG']).astype(str)
                    df['LOC']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'LOC']).astype(str)
                    df['MISC']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'MISC']).astype(str)

                   
                st.write(df)
                # Save data
                csv=convert_df(df)


            if input_ent == "PERSON":
                for i in df.iloc[:,0]:
                    df['PERSON']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'PER']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == "ORGANISATION":
                for i in df.iloc[:,0]:
                    df['ORG']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'ORG']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == "LOCATION":
                for i in df.iloc[:,0]:
                    df['LOC']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'LOC']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == "MISCELLANEOUS":
                for i in df.iloc[:,0]:
                    df['MISC']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'MISC']).astype(str)
                st.write(df)
                # Save data
                csv=convert_df(df)
    else:
        st.write("Sentence : ",input_text)
        df=pd.DataFrame()
        df['Text Input']=input_text
        df['Text Input']=df['Text Input'].astype(str)

        if input_ent == "ALL":
  
            df['PERSON']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'PER']).astype(str)
            df['ORG']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'ORG']).astype(str)
            df['LOC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'LOC']).astype(str)
            df['MISC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'MISC']).astype(str)
            st.write(df)

        if input_ent == "PERSON":
            df['PERSON']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'PER']).astype(str)
            st.write(df)

        if input_ent == "ORGANISATION":
            df['ORG']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'ORG']).astype(str)
            st.write(df)

        if input_ent == 'LOCATION':
            df['LOC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'LOC']).astype(str)
            st.write(df)

        if input_ent == 'MISCELLANEOUS':
            df['MISC']=df['Text Input'].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'MISC']).astype(str)
            st.write(df)
    
def FlanT5_xxl(txt):
    global csv
    global df

    l1=[]
    l2=[]
    l3=[]
    l4=[]
    
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.1})
    #st.write("Model Loaded - ","google/flan-t5-xxl")
    # Accept user questions/query
    person_query = "Mention the person names from the document"
    location_query = "What are locations given in the document?"
    company_query = "What are the companies mentioned in the given document?"
    misc_query = "What is amount involved in the document?"
    
    if document is not None:
        if document.name.endswith('.pdf'):
            st.write("Sentence : ",s)
            df=pd.DataFrame()
            df['Text Input']=pd.Series(s)
            df['Text Input']=df['Text Input'].astype(str)

            if input_ent == "ALL":
                chunks = get_text_chunks(s)
                vectorstore = get_vectorstore(chunks)
                chain = get_conversation_chain(vectorstore)
                doc1 = vectorstore.similarity_search(person_query)
                doc2 = vectorstore.similarity_search(company_query)
                doc3 = vectorstore.similarity_search(location_query)
                doc4 = vectorstore.similarity_search(misc_query)
                l1.append(chain.run(input_documents=doc1, question=person_query))
                l2.append(chain.run(input_documents=doc2, question=company_query))
                l3.append(chain.run(input_documents=doc3, question=location_query))
                l4.append(chain.run(input_documents=doc4, question=misc_query))
                df['PERSON']=l1
                df['ORG']=l2
                df['LOC']=l3
                df['MISC']=l4
                
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == "PERSON":
                chunks = get_text_chunks(s)
                vectorstore = get_vectorstore(chunks)
                chain = get_conversation_chain(vectorstore)
                docs = vectorstore.similarity_search(person_query)
                l1.append(chain.run(input_documents=docs, question=person_query))
                df['PERSON']=l1
                
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == "ORGANISATION":
                chunks = get_text_chunks(s)
                vectorstore = get_vectorstore(chunks)
                chain = get_conversation_chain(vectorstore)
                docs = vectorstore.similarity_search(company_query)
                l1.append(chain.run(input_documents=docs, question=company_query))
                df['ORG']=l1
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == 'LOCATION':
                chunks = get_text_chunks(s)
                vectorstore = get_vectorstore(chunks)
                chain = get_conversation_chain(vectorstore)
                docs = vectorstore.similarity_search(location_query)
                l1.append(chain.run(input_documents=docs, question=location_query))
                df['LOC']=l1
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == 'MISCELLANEOUS':
                chunks = get_text_chunks(s)
                vectorstore = get_vectorstore(chunks)
                chain = get_conversation_chain(vectorstore)
                docs = vectorstore.similarity_search(misc_query)
                l1.append(chain.run(input_documents=docs, question=misc_query))
                df['MISC']=l1
                st.write(df)
                # Save data
                csv=convert_df(df)

        elif document.name.endswith('.txt'):
            st.write("Sentence : ",s)
            df=pd.DataFrame()
            df['Text Input']=pd.Series(s)
            df['Text Input']=df['Text Input'].astype(str)

            if input_ent == "ALL":
                chunks = get_text_chunks(s)
                vectorstore = get_vectorstore(chunks)
                chain = get_conversation_chain(vectorstore)
                doc1 = vectorstore.similarity_search(person_query)
                doc2 = vectorstore.similarity_search(company_query)
                doc3 = vectorstore.similarity_search(location_query)
                doc4 = vectorstore.similarity_search(misc_query)
                l1.append(chain.run(input_documents=doc1, question=person_query))
                l2.append(chain.run(input_documents=doc2, question=company_query))
                l3.append(chain.run(input_documents=doc3, question=location_query))
                l4.append(chain.run(input_documents=doc4, question=misc_query))
                df['PERSON']=l1
                df['ORG']=l2
                df['LOC']=l3
                df['MISC']=l4
                
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == "PERSON":
                chunks = get_text_chunks(s)
                vectorstore = get_vectorstore(chunks)
                chain = get_conversation_chain(vectorstore)
                docs = vectorstore.similarity_search(person_query)
                l1.append(chain.run(input_documents=docs, question=person_query))
                df['PERSON']=l1
                
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == "ORGANISATION":
                chunks = get_text_chunks(s)
                vectorstore = get_vectorstore(chunks)
                chain = get_conversation_chain(vectorstore)
                docs = vectorstore.similarity_search(company_query)
                l1.append(chain.run(input_documents=docs, question=company_query))
                df['ORG']=l1
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == 'LOCATION':
                chunks = get_text_chunks(s)
                vectorstore = get_vectorstore(chunks)
                chain = get_conversation_chain(vectorstore)
                docs = vectorstore.similarity_search(location_query)
                l1.append(chain.run(input_documents=docs, question=location_query))
                df['LOC']=l1
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == 'MISCELLANEOUS':
                chunks = get_text_chunks(s)
                vectorstore = get_vectorstore(chunks)
                chain = get_conversation_chain(vectorstore)
                docs = vectorstore.similarity_search(misc_query)
                l1.append(chain.run(input_documents=docs, question=misc_query))
                df['MISC']=l1
                st.write(df)
                # Save data
                csv=convert_df(df)


        elif document.name.endswith('.csv'):
            
            if input_ent == "ALL":
                for i in df.iloc[:,0]:
                    chunks = get_text_chunks(i)
                    vectorstore = get_vectorstore(chunks)
                    chain = get_conversation_chain(vectorstore)
                    doc1 = vectorstore.similarity_search(person_query)
                    doc2 = vectorstore.similarity_search(company_query)
                    doc3 = vectorstore.similarity_search(location_query)
                    doc4 = vectorstore.similarity_search(misc_query)
                    l1.append(chain.run(input_documents=doc1, question=person_query))
                    l2.append(chain.run(input_documents=doc2, question=company_query))
                    l3.append(chain.run(input_documents=doc3, question=location_query))
                    l4.append(chain.run(input_documents=doc4, question=misc_query))
                df['PERSON']=l1
                df['ORG']=l2
                df['LOC']=l3
                df['MISC']=l4
                st.write(df)
                #Save data
                csv=convert_df(df)


            if input_ent == "PERSON":
                for i in df.iloc[:,0]:
                    chunks = get_text_chunks(i)
                    vectorstore = get_vectorstore(chunks)
                    chain = get_conversation_chain(vectorstore)
                    docs = vectorstore.similarity_search(person_query)
                    l1.append(chain.run(input_documents=docs, question=person_query))
                df['PERSON']=l1
                st.write(df)
                # Save data
                csv=convert_df(df)
                

            if input_ent == "ORGANISATION":
                for i in df.iloc[:,0]:
                    chunks = get_text_chunks(i)
                    vectorstore = get_vectorstore(chunks)
                    chain = get_conversation_chain(vectorstore)
                    docs = vectorstore.similarity_search(company_query)
                    l1.append(chain.run(input_documents=docs, question=company_query))
                df['ORG']=l1
                st.write(df)
                # Save data
                csv=convert_df(df)

            if input_ent == "LOCATION":
                for i in df.iloc[:,0]:
                    chunks = get_text_chunks(i)
                    vectorstore = get_vectorstore(chunks)
                    chain = get_conversation_chain(vectorstore)
                    docs = vectorstore.similarity_search(location_query)
                    l1.append(chain.run(input_documents=docs, question=location_query))
                df['LOC']=l1
                st.write(df)
                # Save data
                csv=convert_df(df)
            
            if input_ent == "MISCELLANEOUS":
                for i in df.iloc[:,0]:
                    chunks = get_text_chunks(i)
                    vectorstore = get_vectorstore(chunks)
                    chain = get_conversation_chain(vectorstore)
                    docs = vectorstore.similarity_search(misc_query)
                    l1.append(chain.run(input_documents=docs, question=misc_query))
                df['MISC']=l1
                st.write(df)
                # Save data
                csv=convert_df(df)
    else:
        st.write("Sentence : ",input_text)
        df=pd.DataFrame()
        df['Text Input']=input_text
        df['Text Input']=df['Text Input'].astype(str)

        if input_ent == "ALL":
            chunks = get_text_chunks(input_text)
            vectorstore = get_vectorstore(chunks)
            chain = get_conversation_chain(vectorstore)
            docs = vectorstore.similarity_search(query)
            l1.append(chain.run(input_documents=docs, question=person_query))
            l2.append(chain.run(input_documents=docs, question=company_query))
            l3.append(chain.run(input_documents=docs, question=location_query))
            l4.append(chain.run(input_documents=docs, question=misc_query))
            df['PERSON']=l1
            df['ORG']=l2
            df['LOC']=l3
            df['MISC']=l4
        
            st.write(df)
            # Save data
            csv=convert_df(df)

        if input_ent == "PERSON":
            chunks = get_text_chunks(input_text)
            vectorstore = get_vectorstore(chunks)
            chain = get_conversation_chain(vectorstore)
            docs = vectorstore.similarity_search(person_query)
            l1.append(chain.run(input_documents=docs, question=person_query))
            df['PERSON']=l1
            
            st.write(df)
            # Save data
            csv=convert_df(df)

        if input_ent == "ORGANISATION":
            chunks = get_text_chunks(input_text)
            vectorstore = get_vectorstore(chunks)
            chain = get_conversation_chain(vectorstore)
            docs = vectorstore.similarity_search(company_query)
            l1.append(chain.run(input_documents=docs, question=company_query))
            df['ORG']=l1
            st.write(df)
            # Save data
            csv=convert_df(df)

        if input_ent == 'LOCATION':
            chunks = get_text_chunks(input_text)
            vectorstore = get_vectorstore(chunks)
            chain = get_conversation_chain(vectorstore)
            docs = vectorstore.similarity_search(location_query)
            l1.append(chain.run(input_documents=docs, question=location_query))
            df['LOC']=l1
            st.write(df)
            # Save data
            csv=convert_df(df)

        if input_ent == 'MISCELLANEOUS':
            chunks = get_text_chunks(input_text)
            vectorstore = get_vectorstore(chunks)
            chain = get_conversation_chain(vectorstore)
            docs = vectorstore.similarity_search(misc_query)
            l1.append(chain.run(input_documents=docs, question=misc_query))
            df['MISC']=l1
            st.write(df)
            # Save data
            csv=convert_df(df)

    #Display the entity visualization in the browser:
    #st.markdown(doc, unsafe_allow_html=True)
    
    #doc = nlp(input_text)

    #Display the entity visualization in the browser:
    #st.markdown(doc, unsafe_allow_html=True)
    #return doc












            
        
# Send the text string to the Roberta nlp object for converting to a 'doc' object.
# Form to accept user's model input for NER


with st.form('Entities Required', clear_on_submit=True):
    options1 = st.selectbox(
    'Choose entity type for NER:',
    options=["ALL", "PERSON", "ORGANISATION", "LOCATION"])
    st.write('You selected:', options1)
    submitted = st.form_submit_button('Submit')
input_ent = options1

result = []
with st.form('NER_form', clear_on_submit=True):
    options = st.selectbox(
    'Choose a model for NER:',
    options=["Roberta", "BERT_base", "FlanT5_xxl"])
    st.write('You selected:', options)
    submitted = st.form_submit_button('Submit')
    response = ''
    if submitted:
        if options=="Roberta":
            with st.spinner('Extracting...'):
                response = Roberta_NER(input_text) 
        elif options=="BERT_base":
            with st.spinner('Extracting...'):
                response = BERT_base(input_text)
        elif options=="FlanT5_xxl":
            with st.spinner("Extracting..."):
                response = FlanT5_xxl(input_text) 

       

        # Download CSV files
#st.download_button( label="Download data as CSV",data=csv,file_name='NER_data.csv',mime='text/csv')


            


#st.info(response)

