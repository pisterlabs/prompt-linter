#!/usr/bin/python

import argparse
import os
import random
import time
import openai

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

# Maximum number of times to retry before giving up.
MAX_RETRIES = 10

# Always retry when these exceptions are raised.
RETRIABLE_EXCEPTIONS = (IOError,)

# Always retry when an HttpError with one of these status codes is raised.
RETRIABLE_STATUS_CODES = [500, 502, 503, 504]

CLIENT_SECRETS_FILE = 'client_secret.json'

SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

VALID_PRIVACY_STATUSES = ('public', 'private', 'unlisted')

def get_authenticated_service():
  flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
  credentials = flow.run_local_server(port=0)
  return build(API_SERVICE_NAME, API_VERSION, credentials=credentials)

def initialize_upload(youtube, options):
  tags = None
  if options.keywords:
    tags = options.keywords.split(',')

  body=dict(
    snippet=dict(
      title=options.title,
      description=options.description,
      tags=tags,
      categoryId=options.category
    ),
    status=dict(
      privacyStatus=options.privacyStatus
    )
  )

  insert_request = youtube.videos().insert(
    part=','.join(body.keys()),
    body=body,
    media_body=MediaFileUpload(options.file, chunksize=-1, resumable=True)
  )

  resumable_upload(insert_request)

def resumable_upload(request):
  response = None
  error = None
  retry = 0
  while response is None:
    try:
      print('Uploading file...')
      status, response = request.next_chunk()
      if response is not None:
        if 'id' in response:
          print('Video id "{}" was successfully uploaded.'.format(response['id']))
        else:
          exit('The upload failed with an unexpected response: {}'.format(response))
    except HttpError as e:
      if e.resp.status in RETRIABLE_STATUS_CODES:
        error = 'A retriable HTTP error {} occurred:\n{}'.format(e.resp.status, e.content)
      else:
        raise
    except RETRIABLE_EXCEPTIONS as e:
      error = 'A retriable error occurred: {}'.format(e)

    if error is not None:
      print(error)
      retry += 1
      if retry > MAX_RETRIES:
        exit('No longer attempting to retry.')

      max_sleep = 2 ** retry
      sleep_seconds = random.random() * max_sleep
      print('Sleeping {} seconds and then retrying...'.format(sleep_seconds))
      time.sleep(sleep_seconds)

if __name__ == '__main__':

  with open("/Users/aiwork/Downloads/temp.txt") as f:
    s = f.read()

    print("temp.txt:")
    print(s)

    parser = argparse.ArgumentParser()
    parser.add_argument('--password', required=True, help='key')
    parser.add_argument('--file', required=True, help='Video file to upload')
    parser.add_argument('--title', help='Video title', default='Test Title')
    parser.add_argument('--description', help='Video description',
      default='Test Description')
    parser.add_argument('--category', default='22',
      help='Numeric video category. ' +
        'See https://developers.google.com/youtube/v3/docs/videoCategories/list')
    parser.add_argument('--keywords', help='Video keywords, comma separated',
      default='')
    parser.add_argument('--privacyStatus', choices=VALID_PRIVACY_STATUSES,
      default='private', help='Video privacy status.')
    args = parser.parse_args()

    youtube = get_authenticated_service()

    openai.api_key = args.password
    prompt = f"""You are the best music producer.
Please create the best title in Youtube for the following music.

# Music Description
{s}"""
    messages = [{"role": "system", "content": prompt}]
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                          messages=messages,
                                          max_tokens=10,
                                          temperature=0.0)
    title = response.choices[0].message.content.strip()
    args.title = title
    print("title:")
    print(title)

    openai.api_key = args.password
    prompt = f"""You are the best music producer.
Please create the best description in Youtube of the music below.

# Music Description
{s}"""
    messages = [{"role": "system", "content": prompt}]
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                          messages=messages,
                                          max_tokens=100,
                                          temperature=0.0)
    description = response.choices[0].message.content.strip()
    args.description = description

    print("description:")
    print(description)

    try:
      initialize_upload(youtube, args)
    except HttpError as e:
      print('An HTTP error {} occurred:\n{}'.format(e.resp.status, e.content))
