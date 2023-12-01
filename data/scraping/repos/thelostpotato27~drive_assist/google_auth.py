from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# import openai
# openai.organization = "org-Q5blmpJY4csYpHCZjCDruM3H"
# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.Model.list()

# pipenv shell

import os

# Scopes define the level of access you need from the user.
SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate_google_drive():
    # If the token.json file exists, load the existing credentials from it.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    else:
        # If no valid credentials are found, start the OAuth flow to obtain them.
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES, redirect_uri='http://localhost:8000/')
        creds = flow.run_local_server(port=8000)

        # Save the credentials for future use.
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    # Check if the credentials have a valid access token or refresh token.
    if not creds.valid:
        # If the credentials are not valid, initiate the OAuth flow to obtain new tokens.
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)

        # Save the new credentials in the token.json file.
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return creds

def retrieve_google_drive_files(creds):
    # Build the service object for calling the Drive API.
    drive_service = build('drive', 'v3', credentials=creds)

    # Call the Drive API to retrieve a list of files with the given search query.
    response = drive_service.files().list(
        q="mimeType='application/vnd.google-apps.document'",
        fields="files(name, id)"
    ).execute()

    # Get the list of files from the response.
    files = response.get('files', [])

    # Print the files.
    if not files:
        print('No files found.')
    else:
        print('Files:')
        for file in files:
            print(f"{file['name']} ({file['id']})")


# def chatgpt_requests(prompt):
#     #this function will be called by the chatgpt app to get the response from the chatgpt api
#     curl https://api.openai.com/v1/chat/completions \
#         -H "Content-Type: application/json" \
#         -H "Authorization: Bearer $OPENAI_API_KEY" \
#         -d {
#             "model": "gpt-3.5-turbo",
#             "messages": [{"role": "user", "content": "Say this is a test!"}],
#             "temperature": 0.7
#         }

def main():
    # Authenticate and get the credentials object.
    creds = authenticate_google_drive()
    
    # Now, you can use the "creds" object to make authorized API requests to Google Drive.
    # For example, you can list files in Google Drive using the Google Drive API.

    retrieve_google_drive_files(creds)





if __name__ == '__main__':
    main()
