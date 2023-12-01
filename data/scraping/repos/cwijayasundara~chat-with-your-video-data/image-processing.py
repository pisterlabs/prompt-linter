import os

from PIL import Image
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

list_images=['images/img.jpg']

image = Image.open(list_images[0]).convert("RGB")
print(image)

llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0.0)

response = llm(image)
print(response.content)