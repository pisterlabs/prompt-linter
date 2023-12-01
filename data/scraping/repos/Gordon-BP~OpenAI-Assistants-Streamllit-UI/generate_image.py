import openai
import os
from util.pydantic_classes import BotImageMessage
from dotenv import load_dotenv

load_dotenv()
client = openai.Client(api_key = os.environ.get('OPENAI_API_KEY'))

def generateImage(prompt:str)->BotImageMessage:
   """
   Uses the new DALLE-3 API to generate an image based

   Args:
    - prompt: What to prompt the image generator with

    Returns:
        A BotImageMessage object containing the URL of the image
   """
   img = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024",
        response_format = "url",
    )
   return BotImageMessage(
      url=img.data[0].url
   )