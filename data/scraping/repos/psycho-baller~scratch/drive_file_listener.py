# filename: drive_file_listener.py

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import GoogleDriveLoader

# If modifying these SCOPES, delete your token.json file.
SCOPES = ["https://www.googleapis.com/auth/drive.metadata.readonly"]
CLIENT_SECRET_FILE = 'token.json'
APPLICATION_NAME = 'Drive API Python Quickstart'

def get_credentials():
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    """Gets valid user credentials from storage."""
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
          token.write(creds.to_json())
    return creds

def main():
    """Shows basic usage of the Google Drive API."""
    try:
        service = build("drive", "v3", credentials=get_credentials())

        # Replace 'root' with your folder's ID
        folder_id = 'root'
        query = f"'{folder_id}' in parents"

        # Store the names of all .txt files in the folder
        txt_files = set()

        results = service.files().list(q=query, pageSize=10, fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])

        for item in items:
            print(item['name'])
            if item['name'] not in txt_files:
                txt_files.add(item['name'])

                # Download the file
                request = service.files().get_media(fileId=item['id'])
                fh = io.FileIO(item['name'], 'wb')
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()

                # Summarize the content
                with open(item['name'], 'r') as file:
                    text = file.read()
                    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
                    chain = load_summarize_chain(llm, chain_type="stuff")

                    sumary = chain.run(docs)
                    print(f"Summary of {item['name']}: {summary}")
    except HttpError as e:
        print(e)
        

if __name__ == '__main__':
    main()
