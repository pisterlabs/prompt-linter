from itsdangerous import base64_decode
import base64 as b64
import streamlit as st
import pandas as pd
from spacy import displacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from langchain.document_loaders import TextLoader

from PyPDF2 import PdfReader


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import os

from langchain import HuggingFaceHub
from langchain.document_loaders import TextLoader
from langchain.document_loaders.image import UnstructuredImageLoader

# Add a section header:
st.set_page_config(page_title="Named Entity Recognition Tagger", page_icon="./logo.png",layout = 'wide')
st.title("📘 Named Entity Recognition")

from streamlit_extras.app_logo import add_logo
add_logo("logo.png", height=60)


# st.text_input takes a label and default text string:
input_text = st.text_input('Text string to analyze:', 'Jennifer is living in New York and has American Express card.')

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



def Roberta_NER(txt):
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


        elif document.name.endswith('.csv'):
            
            if input_ent == "ALL":
                for i in df.iloc[:,0]:
                    df['PERSON']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'PER']).astype(str)
                    df['ORG']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'ORG']).astype(str)
                    df['LOC']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'LOC']).astype(str)
                    df['MISC']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'MISC']).astype(str)

                   
                st.write(df)
                # Save data
                #csv=convert_df(df)


            if input_ent == "PERSON":
                for i in df.iloc[:,0]:
                    df['PERSON']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'PER']).astype(str)
                st.write(df)
                

            if input_ent == "ORGANISATION":
                for i in df.iloc[:,0]:
                    df['ORG']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'ORG']).astype(str)
                st.write(df)

            if input_ent == "LOCATION":
                for i in df.iloc[:,0]:
                    df['LOC']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'LOC']).astype(str)
                st.write(df)
            
            if input_ent == "MISCELLANEOUS":
                for i in df.iloc[:,0]:
                    df['MISC']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'MISC']).astype(str)
                st.write(df)
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


        elif document.name.endswith('.csv'):
            
            if input_ent == "ALL":
                for i in df.iloc[:,0]:
                    df['PERSON']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'PER']).astype(str)
                    df['ORG']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'ORG']).astype(str)
                    df['LOC']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'LOC']).astype(str)
                    df['MISC']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'MISC']).astype(str)

                   
                st.write(df)
                # Save data
                #csv=convert_df(df)


            if input_ent == "PERSON":
                for i in df.iloc[:,0]:
                    df['PERSON']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'PER']).astype(str)
                st.write(df)
                

            if input_ent == "ORGANISATION":
                for i in df.iloc[:,0]:
                    df['ORG']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'ORG']).astype(str)
                st.write(df)

            if input_ent == "LOCATION":
                for i in df.iloc[:,0]:
                    df['LOC']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'LOC']).astype(str)
                st.write(df)
            
            if input_ent == "MISCELLANEOUS":
                for i in df.iloc[:,0]:
                    df['MISC']=df.iloc[:,0].apply(lambda sent: [ent['word'] for ent in nlp(sent) if ent['entity_group'] == 'MISC']).astype(str)
                st.write(df)
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
    options=["Roberta", "BERT_base"])
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
       

        # Download CSV files
        #st.download_button( label="Download data as CSV",data=df,file_name='NER_data.csv',mime='text/csv')
        # Assume my data is present in df
        csv = df.to_csv(index=False)
        b164 = b64.b64encode(csv.encode()).decode()
        st.markdown(f'<a href="data:file/csv;base64;{b164}" download="results.csv">Download Results</a>', unsafe_allow_html=True)

            


#st.info(response)

