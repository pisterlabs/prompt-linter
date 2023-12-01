import streamlit as st
# from invoice_backend import *
from PIL import Image
import json
from langchain.llms import OpenAI
from PyPDF2 import PdfReader
import pandas as pd
import re
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
import os
import regex as re
# import re
import csv
import json
# import pyodbc
import mysql.connector
from mysql.connector import Error
import streamlit as st
# os.environ["OPENAI_API_KEY"] = "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"
key = st.secrets['key']


def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
def extract_data(pages_data):

    template = '''Extract all following values: invoice no., Date, Shipmode,Ship to, Bill to, Balance Due, Item,  
    Quantity, Rate, discount ,Shipping, Total,
    Order ID from this data{pages}. If the word Discount is not in the page return 0.0 else return the value for it. Note that the output can have multiple records with same invoice number if there are more than 1 row of Items in the invoice.
    
    Expected output : Remove dollar sign if found anywhere in {pages} and return the final answer in this format {{'Invoice no.':1001329, 
    'Item':'Office Chair', 'Quantity':2, 'Date':'05/01/2022', 
    'Rate':1100.00, 'Discount' : 220, 'mount':2200.00, 'Sub Total':1980.00, 'Balance Due':1980.00,
    'Ship Mode':'First Class','Ship to':'Saltillo, Coahuila,Mexico', 'Order ID':'MX-2012-AH1003082-41251'}}
    '''

    prompt_template = PromptTemplate(input_variables=['pages'], template=template)

    llm = OpenAI(temperature=0.4,openai_api_key=key)
    full_response = llm(prompt_template.format(pages=pages_data))

    return full_response
def extract_df(extracted_text):
    pattern = r'{(.+)}'
    match = re.search(pattern, extracted_text, re.DOTALL)
    if match:
        extracted_data = match.group(1)
        # Converting the extracted text to a dictionary
        data_dict = eval('{' + extracted_data + '}')
        df_temp=pd.DataFrame([data_dict])
        df_temp=df_temp.set_index('Order ID').reset_index()
        return df_temp
    else:
        return 'Could Not extract the Data'



def ingest_to_db_sat(df):
    # try:
    ik = df.to_json(orient='records')
    json_string = json.loads(ik)
    d1= json_string[0]
    # st.write(d1)
    import mysql.connector
    from mysql.connector import Error
    # Establish a database connection
    db_connection = mysql.connector.connect(
        host="sqldatabase.mysql.database.azure.com",
        user="yusuf121",
        password="Satwik121",
        database="chatbotdb"
    )
    cursor = db_connection.cursor()
    # st.write("here is") 

    insert_query = """INSERT INTO invoice(`Order ID`, `Invoice no.`, `Date`, `Ship Mode`, `Ship to`, `Bill to`,
    `Balance Due`, `Item`, `Quantity`, `Rate`, `Discount`, `Shipping`, `Total`)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    cursor.execute(insert_query, (
                    d1.get("Order ID", None),
                    d1.get("Invoice no.", None),
                    d1.get("Date", None),
                    d1.get("Ship Mode", None),
                    d1.get("Ship to", None),
                    d1.get("Bill to", None),
                    d1.get("Balance Due", None),
                    d1.get("Item", None),
                    d1.get("Quantity", None),
                    d1.get("Rate", None),
                    d1.get("Discount", None),
                    d1.get("Shipping", None),
                    d1.get("Total", None)

                ))

    db_connection.commit()
    return "Ingested Successfully"

    # except Error as e:
    #     st.error(f"Error connecting to the database: {str(e)}")
    #     return "Failed to connect DB"


#---------------------------------------------------------------------------main----------------------------------------------------------------
def app():
    # st.set_page_config(page_title='Invoice Reader',page_icon='.\Data\invoice.png')
    with st.container():
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.image('dbot.png', width=75,)
                #st.image('./data/2.png', width=300)
            with col2:
                st.title('Invoice Reader')
    with st.sidebar:
        # col1, col2, col3 = st.columns(3)
        # with col2:
        image=Image.open('invoice.png')
        new_image = image.resize((200, 200))
        st.image(new_image,use_column_width='auto',width=300)
        st.header('📜Invoice Reader')
        st.write('Use our OpenAI & Langchain powered invoice reader to fetch the data in tabular form, ingest it to your database and download it in just a few steps. ')
        st.warning('The results may not be always reliable',icon='🚨')
    uploaded_file = st.file_uploader("Upload your document here 👇", type=["pdf"])
    file_uploaded=False
    
    if uploaded_file:
        
        raw_data=get_pdf_text(uploaded_file)
        extracted_text=extract_data(raw_data)
        df=extract_df(extracted_text)
        # ik = df.to_json(orient='records')
        # json_string = json.loads(ik)
        # d1= json_string[0]
        # st.write(d1)
        if type(df)== pd.core.frame.DataFrame:
            ingest_button=st.button('Ingest Data')  #comment
            st.dataframe(df)
            
            if ingest_button:
                 x=ingest_to_db_sat(df)
                 st.markdown(x)
        else:
            st.write("no pdf detected")
            st.markdown(df)
    else:
         st.markdown('Please Upload an invoice pdf.')


