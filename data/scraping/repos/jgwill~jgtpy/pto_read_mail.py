
import os
import base64
import re
import openai
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def read_emails():
  creds = None
  # The file token.pickle stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
  if os.path.exists('token.pickle'):
    with open('token.pickle', 'rb') as token:
      creds = pickle.load(token)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:

      home_dir = os.path.expanduser("~")
      credentials_path = os.path.join(home_dir, '.gcp-gaiaapp231117-web.json')

      flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open('token.pickle', 'wb') as token:
      pickle.dump(creds, token)

  service = build('gmail', 'v1', credentials=creds)

  # Call the Gmail API
  results = service.users().messages().list(userId='me').execute()
  messages = results.get('messages', [])

  for message in messages:
    msg = service.users().messages().get(userId='me', id=message['id']).execute()
    email_data = msg['payload']['headers']
    for values in email_data:
      name = values['name']
      if name == 'From':
        from_name = values['value']
        for part in msg['payload']['parts']:
          try:
            data_text = part['body']["data"]
            data_text = base64.urlsafe_b64decode(data_text).decode()
            data = data_text.replace("\n", "")
            summarize_email(data)
          except BaseException:
            pass

def summarize_email(email_text):
  prompt_text = f"Summarize the following email: {email_text}"
  completion = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt_text,
    temperature=0.5,
  )
  print(completion.choices[0].text.strip() if completion.choices else "")

read_emails()