import requests
import time
import requests
import json
from io import BytesIO
import json
import os
# from tika import parser
import base64
import pandas as pd
import openai
# import pdftotext
import math
from langchain import PromptTemplate, LLMChain
import time
import textract
import pandas as pd
import pymongo
from bson.objectid import ObjectId
from PIL import Image
# from database.ConnectDb import DatabaseHandler
from datetime import datetime
# from cloudStorage import AzureBlobStorage, save_to_cloud_storage

# from azure.storage.blob import BlobServiceClient
# Authentication configuration
client_id = "09aacaa6-422b-4dcc-9a2f-fc39644f4b32"
client_secret = "SvM8Q~WXrMGl~gvHtkJZkydx.O7UeeD-LA2htdav"
scope = "https://graph.microsoft.com/Mail.Read https://graph.microsoft.com/Mail.ReadWrite https://graph.microsoft.com/Mail.Read.Shared https://graph.microsoft.com/Mail.ReadWrite.Shared https://graph.microsoft.com/Files.Read https://graph.microsoft.com/Files.ReadWrite"
username = "demo@algo8.ai"
password = "AI@algo8"

# db_sql = DatabaseHandler()

openai.api_key = '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'

dbcolumns = ['timestamp','receivedDateTime', 'sender', 'receivedOn', 'subject', 'documentTitle', 'po', 'invoicedFrom', 'invoicedTo', 'invoiceDate','dueDate', 'taxId', 'totalBeforeTax','tax', 'totalAfterTax','currency', 'reject','validate','markedForReview','email','invoicePdf','status']
def get_access_token():
    # Token request configuration
    token_url = "https://login.microsoftonline.com/08b7cfeb-897e-469b-9436-974e694a8df2/oauth2/v2.0/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": scope,
        "grant_type": "password",
        "username": username,
        "password": password
    }

    # Send token request
    response = requests.post(token_url, headers=headers, data=data)
    response.raise_for_status()

    # Extract and return the access token
    access_token = response.json().get("access_token")
    return access_token

def fetch_emails(filter_params):
    # Microsoft Graph API endpoint for fetching emails
    graph_api_endpoint = "https://graph.microsoft.com/v1.0/users/demo@algo8.ai/messages"

    # Fetch emails using Microsoft Graph API
    while True:
        access_token = get_access_token()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        # Define the parameters for fetching emails
        params = {
            "$select": "sender,receivedDateTime,subject,hasAttachments",
            "$filter": filter_params
        }

        # Fetch emails using Microsoft Graph API
        response = requests.get(graph_api_endpoint, headers=headers, params=params)
        if response.status_code == 401:  # Token expired
            time.sleep(1)  # Sleep for a second to avoid excessive token requests
            continue
        response.raise_for_status()
        emails = response.json().get("value", [])
        return emails
    
def pipeline(start_date_filter,end_date_filter, start_time_filter, end_time_filter):
    try:
        graph_api_endpoint = "https://graph.microsoft.com/v1.0/users/demo@algo8.ai/messages"
        access_token = get_access_token()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        print("Code Started")
        # Combine date and time filters
        if start_date_filter and start_time_filter and end_time_filter:
            start_datetime = datetime.combine(start_date_filter, start_time_filter)
            end_datetime = datetime.combine(end_date_filter, end_time_filter)
            start_datetime_str = start_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_datetime_str = end_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')
            date_time_filter = f"receivedDateTime ge {start_datetime_str} and receivedDateTime le {end_datetime_str}"
        else:
            date_time_filter = ''

        # Fetch emails based on filters
        filter_params = []
        if date_time_filter:
            filter_params.append(date_time_filter)
        if filter_params:
            filter_query = ' and '.join(filter_params)
            print(filter_query)
            # Fetch the emails and their attachments
            try:
                page_endpoint = f'{graph_api_endpoint}?$filter={filter_query}&$top=1000'
                response = requests.get(page_endpoint, headers=headers)
                emails = response.json()['value']
            except Exception as e:
                print("error occured",e)
            table_data = []  # List to store the table rows

            t1 = time.time()
            try:
                # Create a Streamlit table to display the DataFrame
                for idx, email in enumerate(emails):
                    email_data = []  # List to store email-related data for each row
                    dt = datetime.now()
                    print(email)
                    timestamp_data = dt.date()
                    email_id = email['id']
                    sender = email['sender']['emailAddress']['address']
                    received_date = email['receivedDateTime']
                    sentDateTime = email['sentDateTime']
                    subject = email['subject']
                    has_attachments = email['hasAttachments']
                    print(sender,subject,has_attachments,received_date)
                    # email_data.extend([sender, received_date, subject, has_attachments])
                    email_data.extend([timestamp_data,received_date,sender,username,subject])
                    
            except Exception as e:
                print("Error Occured! " ,e)
            df = pd.DataFrame(table_data,columns=dbcolumns)

            df1 = df.copy()
            # columns = ['receivedDateTime','sender',	'receivedOn',	'subject'	,'documentTitle',	'po',	'invoicedFrom',	'invoicedTo',	'invoiceDate','dueDate', 'taxId','totalBeforeTax']	
            new_column_names = {'receivedDateTime':'Received Date and Time','sender':'Sender','receivedOn':'Received On','subject':'Subject','documentTitle':'Document Title','po':'PO','invoicedFrom':'Invoiced From','invoicedTo':'Invoiced To','invoiceDate':'Invoice date','dueDate':'Due date','taxId':'Tax ID','totalAfterTax': 'Invoice amount'}
            df1.rename(columns=new_column_names, inplace=True)
            columns = ['Received Date and Time' , 'Sender', 'Received On',  'Subject', 'Document Title', 'PO' ,'Invoiced From', 'Invoiced To', 'Invoice date','Due date','Tax ID', 'Invoice amount']
            df1 = df1[columns]

            # db_sql.run_df(df)
            print("Pushed Data To Database")
            return True
        else:
            print("No Email Found")
            return False
    except Exception as e:
        print("Error Occured",e)

connection_string = "DefaultEndpointsProtocol=https;AccountName=algo8;AccountKey=H97z2x/0YTnCsfSgobknYfGs1jKAjgc9UNsGVrgvzEE3utFs3ENFJmKCLGpWWJtruN7bW0YIqU85+AStmyaPDQ==;EndpointSuffix=core.windows.net"

# blob_service_client = BlobServiceClient.from_connection_string(connection_string)
startDate = "2023-10-23"
startTime = "03:00:00"
endDate = "2023-10-24"
endTime = "23:59:00"

start_date_filter = datetime.strptime(startDate, '%Y-%m-%d').date()
end_date_filter = datetime.strptime(endDate, '%Y-%m-%d').date()

# Convert start_time_filter and end_time_filter to datetime.time objects
start_time_filter = datetime.strptime(startTime, '%H:%M:%S').time()
end_time_filter = datetime.strptime(endTime, '%H:%M:%S').time()
bot_pipeline = pipeline(start_date_filter, end_date_filter, start_time_filter, end_time_filter)

print(bot_pipeline)