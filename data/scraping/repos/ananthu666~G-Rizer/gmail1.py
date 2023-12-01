from __future__ import print_function
import os.path

from googleapiclient import errors 
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
import email

import openai
import requests
import json
def sum(text):
    openai.api_key = 'sk******************************************'
    URL = "https://api.openai.com/v1/chat/completions"

    payload = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": " give heading :  and summarize:  with number of words less than 150"+text}],
    "temperature" : 1.0,
    "top_p":1.0,
    "n" : 1,
    "stream": False,
    "presence_penalty":0,
    "frequency_penalty":0,
    }

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai.api_key}"
    }

    response = requests.post(URL, headers=headers, json=payload, stream=False)

    data=json.loads(response.content)

    summary = data['choices'][0]['message']['content']
    # print(summary)
    # print(summary.split('\n\n')[0])
    # print(summary.split('\n\n')[1])
    return summary


SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


def get_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)    
    return service


def search_message(service, user_id, search_string):

    try:
        list_ids = []

        search_ids = service.users().messages().list(userId=user_id, q=search_string).execute()
        try:
            ids = search_ids['messages']
        except KeyError:
            print("WARNING: the search queried returned 0 results")
            print("returning an empty string")
            return ""

        if len(ids)>1:
            for msg_id in ids:
                list_ids.append(msg_id['id'])
            return(list_ids)

        else:
            list_ids.append(ids[0]['id'])
            return list_ids
        
    except errors.HttpError as e:
         print(f"An error occurred: {e}")


def get_message(service, user_id, msg_id):
    try:
        
        message = service.users().messages().get(userId=user_id, id=msg_id,format='raw').execute()
        # print("message",message)
        
        msg_str = base64.urlsafe_b64decode(message['raw'].encode('ASCII'))
        # print(type(msg_str))
        
        mime_msg = email.message_from_bytes(msg_str)
        # print("mime_msg",mime_msg)
        
        content_type = mime_msg.get_content_maintype()
        # print('here')
        if content_type == 'multipart':

            parts = mime_msg.get_payload()

            final_content = parts[0].get_payload()
            return final_content

        elif content_type == 'text':
            return mime_msg.get_payload()

        else:
            return ""
            print("\nMessage is not text or multipart, returned an empty string")
    except Exception:
         print(f"An error occurred")



def enable(mail):
    service1=get_service()
    # print(service1)
    service2=search_message(service1,'me',mail)
    # print(service2[0])
    service3=get_message(service1,'me',service2[0])
    # print(service3)
    return sum(service3)

def my_function(email1, email2):
    # Your function logic here
    paragraph=enable(email2)
    paragraph_text = paragraph
    return paragraph_text

# x=my_function("email1", "email2")
# print(x)