import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import json
import mysql.connector
from mysql.connector import Error
import pandas as pd
from decouple import config
from dotenv import load_dotenv
import os
import openai
import config 

# api_key = os.getenv("api_key")
# endpoint = os.getenv("endpoint")

api_key = st.secrets['api_key']
endpoint = st.secrets['endpoint']

print(api_key)
print(endpoint)

model_id = "receipt_model"
#formUrl = "YOUR_DOCUMENT"  

# Create a Form Recognizer client
#form_recognizer_client = FormRecognizerClient(endpoint, AzureKeyCredential(api_key))
document_analysis_client = DocumentAnalysisClient(
    endpoint=endpoint, credential=AzureKeyCredential(api_key)
)

dict1 = {}
# Create a Streamlit app
st.title("Reciept Extractor")

# Upload PDF file using Streamlit
uploaded_file = st.file_uploader("Upload an Invoice or Receipt PDF", type=["jpg","png"])
from io import BytesIO

# Read the contents of the uploaded file
try:
    if uploaded_file:
        file_contents = uploaded_file.read()
        #file_contents = uploaded_fil
        # Create a file stream using BytesIO
        file_stream = BytesIO(file_contents)
except Error as e:
    st.error(f"Upload the File ")


if uploaded_file is not None:
    with uploaded_file:
        st.write("File Uploaded! Analyzing...")
        file_contents = uploaded_file.read()
        # Analyze the content of the document
        poller =document_analysis_client.begin_analyze_document( model_id = model_id, document= file_stream)

        documents = poller.result()

        # Display extracted entities
        for idx, document in enumerate(documents.documents):
            st.subheader(f"Document #{idx + 1} Entities:")
            
            for name, field in document.fields.items():
                dict1[name] = field.value

        
        dict1.pop("prod table")
        
        #st.write(dict1)
        

        import pandas as pd
        # Create a list of indices
        index = list(range(1, 2))
        df = pd.DataFrame(dict1,index = index)
        #st.write(df)
        df.columns = ['VendorOrg', 'ClientName','Subtotal', 'Total', 'Tax', 'VendorAddress', 'ClientAddress', 'ShippingAddress', 'Receipt', 'ReceiptDate', 'DueDate', 'PONumber']
        #df.to_csv('rcpt.csv',index = False)
        ik = df.to_json(orient='records')
        json_string = json.loads(ik)
        d1= json_string[0]
        st.write(d1)
        st.write(df)

        try:
            # Establish a database connection
            db_connection = mysql.connector.connect(
                host="sqldatabase.mysql.database.azure.com",
                user="yusuf121",
                password="Satwik@121",
                database="chatbotdb"
            )
            cursor = db_connection.cursor()

            insert_query = """INSERT INTO receipt (VendorOrg, ClientName, Subtotal, Total, Tax, VendorAddress, ClientAddress, ShippingAddress, Receipt, ReceiptDate, DueDate, PONumber) 
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
              
            cursor.execute(insert_query, (
                d1.get("VendorOrg", None), 
                d1.get("ClientName", None), 
                d1.get("Subtotal", None), 
                d1.get("Total", None), 
                d1.get("Tax", None), 
                d1.get("VendorAddress", None), 
                d1.get("ClientAddress", None), 
                d1.get("ShippingAddress", None), 
                d1.get("Receipt", None), 
                d1.get("ReceiptDate", None), 
                d1.get("DueDate", None), 
                d1.get("PONumber", None)
            ))

            db_connection.commit()

        except Error as e:
            st.error(f"Error connecting to the database: {str(e)}")

        st.write(" Details Added successfully in the table ")
        db_connection.commit()



        
        
        






