import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import json
import mysql.connector
from mysql.connector import Error
import pandas as pd

import os
import openai
from io import BytesIO

from PIL import Image
# import config 

# api_key = os.getenv("api_key")
# endpoint = os.getenv("endpoint")

api_key = st.secrets['api_key']
endpoint = st.secrets['endpoint']

# api_key = "33a331f9eb4c4d718f3557a817ee55b0"
# endpoint = "https://shyam-ocr-123.cognitiveservices.azure.com/"


def rcpt1():

    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    # print(api_key)
    # print(endpoint)

    model_id = "receipt_model"
    #formUrl = "YOUR_DOCUMENT"  

    # Create a Form Recognizer client
    #form_recognizer_client = FormRecognizerClient(endpoint, AzureKeyCredential(api_key))
    document_analysis_client = DocumentAnalysisClient(
        endpoint=endpoint, credential=AzureKeyCredential(api_key)
    )

    dict1 = {}
    # Create a Streamlit app
    # st.title("Receipt Extraction App")
    with st.container():
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.image('./data/Home.png', width=300)
                #st.image('./data/2.png', width=300)
            with col2:
                st.markdown("""
                    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 35vh;">
                        <h1> Welcome To Reciept Extraction App!</h1>
                        <!-- Additional content goes here -->
                    </div>
                """, unsafe_allow_html=True)
    st.markdown("This App is designed to help you extract information from your receipts and store it in a real-time database.")
    # Key Features
    st.header("Key Features")
    st.markdown("""
    - **Image Upload**: Upload a receipt image in JPG or PNG format.
    - **Receipt Extraction**: Extract information from the uploaded image.
    - **Data Storage**: Store the extracted data in a real-time database.
    - **User-Friendly**: A simple and user-friendly interface for easy use.
    """)

    st.header("How to Use")
    st.markdown("""
    1. Click the "Upload Receipt Image" button.
    2. Select a JPG or PNG image of your receipt.
    3. Click the "Extract and Store Receipt Info" button to initiate extraction and storage.
    4. The extracted information will be stored in the database.

    **Note**: The quality of the extracted information depends on the image quality and receipt layout.
    """)

    with st.sidebar:
        st.session_state.uploaded_file = st.file_uploader("Upload a Receipt in jpg/png Format", type=["jpg","png"])

        # st.image('./data/Home.png', width=300)

        st.warning('Model is Trained on Specific Templates Only', icon="⚠️")

    # Upload PDF file using Streamlit




    if st.session_state.uploaded_file is not None:
        with st.session_state.uploaded_file:
            st.write("File Uploaded! Analyzing...")
            # img = Image.open(st.session_state.uploaded_file)
            # # Resize the image to your desired dimensions (e.g., 400x400 pixels)
            # img = img.resize((300, 400))    
            # st.image(st.session_state.uploaded_file, caption="Uploaded Image", use_column_width=True)
            # st.image(img, caption="Uploaded Image", use_column_width=True)

            file_contents = st.session_state.uploaded_file.read()

            
            file_stream = BytesIO(file_contents)
            
            # Analyze the content of the document
            poller =document_analysis_client.begin_analyze_document( model_id = model_id, document= file_stream)

            documents = poller.result()

            # Display extracted entities
            for idx, document in enumerate(documents.documents):
                st.subheader(f"Document #{idx + 1} Entities:")
                
                for name, field in document.fields.items():
                    dict1[name] = field.value
            
            #---------------------------------------------------------------Product Table------------------------------------------------

            # tbl = dict1.get("prod table")
            # import re
            # def extract_info(text):
            #     info = re.findall(r"'(.*?)': DocumentField\(value_type=string, value='(.*?)'", text)
            #     return {key: value for key, value in info}

            # # Extract the information from each string
            # extracted_data = [extract_info(text) for text in tbl]

            # # Create a DataFrame
            # df = pd.DataFrame(extracted_data)
            # st.write(df)

            dict1.pop("prod table")
            
            

            
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
            
            
            # d1 = {
            #     "VendorOrg":"East Repair Inc.",
            #     "ClientName":"John Smith",
            #     "Subtotal":"145.00",
            #     "Total":"$154.06",
            #     "Tax":"9.06",
            #     "VendorAddress":"1912 Harvest Lane New York, NY 12210",
            #     "ClientAddress":"2 Court Square New York, NY 12210",
            #     "ShippingAddress":"John Smith 3787 Pineview Drive Cambridge, MA 12210",
            #     "Receipt":"US-001",
            #     "ReceiptDate":"11/02/2019",
            #     "DueDate":"26/02/2019",
            #     "PONumber":"2312/2019",
            #     }
            # st.write(d1)


            if st.button("Download :memo:"):
                import base64
                # Convert the JSON data to a JSON string
                json_data = json.dumps(d1, indent=4)
                
                # Create a data URI for the JSON string
                b64 = base64.b64encode(json_data.encode()).decode()
                href = f'<a href="data:application/json;base64,{b64}" download="data.json">Download JSON</a>'
                st.markdown(href, unsafe_allow_html=True)

            st.write(df)
            but1 = st.button("Ingest To The Database")
            if but1:

                try:
                    import mysql.connector
                    # Establish a database connection
                    db_connection = mysql.connector.connect(
                        host="sqldatabase.mysql.database.azure.com",
                        user="yusuf121",
                        password="Satwik121",
                        database="chatbotdb"
                    )
                    cursor = db_connection.cursor()
                    # st.write("here is") 

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
                    st.write(" Details Added Successfully in the Table  ")

                except Error as e:
                    st.error(f"Error connecting to the database: {str(e)}")

                
            # db_connection.commit()



        
        
        






