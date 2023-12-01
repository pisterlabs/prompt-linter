import os
import pickle
import google.auth
import google.auth.transport.requests
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from getTodayMails import get_today_emails
from cohereAPI import classifyData, summarizeData, generateData
# Scopes required for accessing Gmail API
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Path to store the credentials
TOKEN_PATH = 'token.pickle'

def authenticate():
    creds = None

    # Check if token file already exists
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, 'rb') as token:
            creds = pickle.load(token)

    # If token is invalid or doesn't exist, generate a new one
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(google.auth.transport.requests.Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=8080)
        
        # Save the credentials for future use
        with open(TOKEN_PATH, 'wb') as token:
            pickle.dump(creds, token)

    # Build and return the Gmail service
    service = build('gmail', 'v1', credentials=creds)
    return service

# Example usage
def main():
    service = authenticate()
    emailData = get_today_emails(service)

    generateData('How is Deep learning different from Machine Learning?')

    response = classifyData(emailData)

    with open('response.txt', 'w') as file:
    # Write each index to a new line in the file
        for index in emailData:
            file.write(str(index) + '\n')
    
    ham_indexes = []
    for index, classification in enumerate(response):
        if classification.prediction == 'ham':
            ham_indexes.append(index)
    with open('ham_indexes.txt', 'w') as file:
    # Write each index to a new line in the file
        for index in ham_indexes:
            file.write(str(index) + '\n')
    print(ham_indexes)


    results = service.users().labels().list(userId='me').execute()
    labels = results.get('labels', [])

    if not labels:
        print('No labels found.')
    else:
        print('Labels:')
        for label in labels:
            print(label['name'])

if __name__ == '__main__':
    main()
