from email.message import EmailMessage
import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import openai
import base64
from email import message_from_string 



def authenticate_gmail_api(credentials_file, token_file):
    creds = None

    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, ['https://www.googleapis.com/auth/gmail.readonly'])
            creds = flow.run_local_server(port=0)

        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)

    return creds



def read_email_message(creds):

    service = build('gmail', 'v1', credentials=creds)

    try:
        response = service.users().messages().list(userId='kolappan@gmail.com', maxResults=1).execute()
        message_id = response['messages'][0]['id']
        message = service.users().messages().get(userId='kolappan@gmail.com', id=message_id).execute()
        text = message['snippet']
    except HttpError as error:
        print(f'An error occurred: {error}')


def read_latest_email(creds):
    try:

        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=1).execute()
        messages = results.get('messages', [])

        if not messages:
            print('No messages found.')
        else:
            for message in messages:
                msg = service.users().messages().get(userId='me', id=message['id'], format='raw').execute()
                msg_str = base64.urlsafe_b64decode(msg['raw'].encode('ASCII')).decode('utf-8')
                mime_msg = message_from_string(msg_str)
                # print(f"Subject: {mime_msg['subject']}")
                # print(f"Sender: {mime_msg['from']}")
                #  print(msg_str)

                if mime_msg.is_multipart():
                    for part in mime_msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True)
                            # print(f"Body: {body.decode('utf-8')}")
                            break
                else:
                    body = mime_msg.get_payload(decode=True)
                    # print(f"Body: {body.decode('utf-8')}")

    except HttpError as error:
        print(f"An error occurred: {error}")
    return(body.decode('utf-8'))


def read_unread_emails(creds):
    service = build('gmail', 'v1', credentials=creds)
    results = service.users().messages().list(userId='me', labelIds=['INBOX'], q="is:unread", maxResults=1).execute()
    messages = results.get('messages', [])

    email_texts = []

    if not messages:
        print('No unread messages found.')
    else:
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id'], format='raw').execute()
            msg_str = base64.urlsafe_b64decode(msg['raw'].encode('ASCII')).decode('utf-8')
            mime_msg = message_from_string(msg_str)

            if mime_msg.is_multipart():
                for part in mime_msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True)
                        email_texts.append(body.decode('utf-8'))
                        break
            else:
                body = mime_msg.get_payload(decode=True)
                email_texts.append(body.decode('utf-8'))

    return email_texts


def gpt_understand_email(api_key, email_text):
    openai.api_key = api_key

    # prompt = f"Please summarize and understand the following email message: \n\n{email_text}\n\nSummary and Understanding:"
    # prompt = f"Please summarise the message and find out product name, price, quantity. \n\n{email_text}\n\nEach line contains an offer"
    # prompt = f"Find out the product name , price and quantity  from the text for ALL the products mentioned in the text. Write one line per product. For each product, write product name, product price, product quanity with labels in that line. \n\n{email_text}\n\n"
    prompt = f"Find out the product name , price and quantity  from the text for ALL the products mentioned in the text. \n\n{email_text}\n\n"

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.5,
    )

    return response.choices[0].text.strip()

credentials_file = 'clientSecret.json'
token_file = 'token.pickle'
creds = authenticate_gmail_api(credentials_file, token_file)

if __name__ == '__main__':
    read_latest_email(creds)


api_key = ""
message_id = '0.1.BB.FDD.1D95BB60F5BE61C.0@omptrans.email.mynrma.com.au'
# email_text = read_email_message(creds)
# print(email_text)
# email_text = read_latest_email(creds)
email_texts = read_unread_emails(creds)
for email_text in email_texts:
    understanding = gpt_understand_email(api_key, email_text)
    # print("Email Text:")
    # print(email_text)
    print("Understanding:")
    print(understanding)
    print("\n")
