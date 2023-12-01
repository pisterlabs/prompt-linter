from django.http.response import JsonResponse
from rest_framework.response import Response
import logging
from .serializer import *
from logtail.handler import LogtailHandler
from datetime import datetime
import openai
import weaviate
import os
from memeModel import idl

import requests
import base64

# handler = LogtailHandler(source_token="tvoi6AuG8ieLux2PbHqdJSVR")
# logger = logging.getLogger(__name__)
# logger.handlers = []
# logger.addHandler(handler)
# logger.setLevel(logging.INFO)

WEAVIATE_URL = os.getenv('WEAVIATE_URL')
CLIENT_CONFIG = None

if WEAVIATE_URL:
    # If a URL is provided, assume this is our remote instance
    # and check for an access token.
    access_token = os.getenv('ACCESS_TOKEN')
    if not access_token:
        raise ValueError("ACCESS_TOKEN environment variable not set. " \
            "This is needed to log into the Weaviate instance.")
    CLIENT_CONFIG = weaviate.AuthBearerToken(
        access_token=access_token,
        expires_in=300 # this is in seconds, by default 60s
        )
else:
    WEAVIATE_URL = 'http://localhost:8080'

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required for uploading objects to Weaviate")


CLIENT = weaviate.Client(WEAVIATE_URL,
                         auth_client_secret=CLIENT_CONFIG,
                         additional_headers={"X-OpenAI-Api-Key": OPENAI_API_KEY})

CLIENT.batch.configure(
    batch_size=10,
    dynamic=True,
    timeout_retries=3,
    callback=None
    )

def hello_world(helloWorldRequest):
  """
    Demo function for testing purposes
  """
  # logger.info(helloWorldRequest)
  # logger.info(helloWorldRequest.name)


def match_text_to_meme(matchTextToMemeRequest):
  """
    Generate meme text for a given input text
  """

  try:
    emotionalDescription = "Describe the emotional value of this text " + str(matchTextToMemeRequest.InputText)

    memeQueryText = openai.Completion.create(
      engine="text-davinci-003",
      prompt=emotionalDescription,
      temperature=0.7,
      max_tokens=500,
      n=1,
      stop=None,
      frequency_penalty=0,
      best_of=10,
      presence_penalty=0,
    )

    # logger.info("Meme query text: " + str(memeQueryText))
    print("Meme query text: " + str(memeQueryText))

    # Use that text to match with memes
    source_text = { "concepts": memeQueryText.choices[0].text }

    weaviate_results = CLIENT.query.get("Meme", [
       'description', 'image', 'template_url']).with_near_text(source_text).with_limit(10).do()

    # logger.info("Weaviate results: " + str(weaviate_results))

    # Return the meme URL and description
    return MatchTextToMemeResponse(
      Memes=weaviate_results["data"]["Get"]["Meme"],
      Error=None,
    )

  except Exception as e:
    return MatchTextToMemeResponse(
      Memes=[],
      Error=str(e),
    )



def index_memes_weaviate(indexMemesWeaviateRequest):
  """
    Index memes in weaviate
  """

  image_desc_pairs = \
    zip(indexMemesWeaviateRequest.MemeImageLink,
        indexMemesWeaviateRequest.Description)

  for url, description in image_desc_pairs:
    response = requests.get(url, allow_redirects=True)
    b64_encoded_image = base64.b64encode(response.content).decode('utf-8')

    data_object = {
      'name': description.replace(' ', '-'),
      'template_url': url,
      'description': description,
      'image': b64_encoded_image
      }

    # logger.info(f"Uploading meme found at {url} " \
    #             f"with the following description: {description}")

    # Write image data to file to check if images are encoded correctly.
    # with open(f'image-uploaded-to-weaviate-{i}.png', 'wb') as f:
    #     f.write(base64.b64decode(b64_encoded_image))

    try:
      CLIENT.batch.add_data_object(data_object, "Meme")
      print(f'Successfully added Meme object {description} to batch!')
    except Exception as e:
       return IndexMemesWeaviateResponse(Error=str(e))

  # logger.info('Sending Meme objects to Weaviate')
  CLIENT.batch.flush()
  return IndexMemesWeaviateResponse(Error=None)
