# (C) Heikki Kupiainen 2023    
# A utility to interact with OpenAI

import os

import openai
import requests 

class MSOpenai(object):

    @staticmethod
    def create_image(prompt):
      openai.api_key = os.getenv("OPENAI_API_KEY")
      response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024",
      )
      url = response["data"][0]["url"]
      print(url)
      MSOpenai.download_image(prompt.replace(" ", "-"), url)
      return response
    
    @staticmethod
    def download_image(name, url):
      img_data = requests.get(url).content
      with open(f"{name}.png", 'wb') as handler:
        handler.write(img_data)

def test_create_image():
  response = MSOpenai.create_image("rocket science")
  print(response["data"][0]["url"])