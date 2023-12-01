import os
import openai
from chatbot_settings import ChatBotSettings

i = ChatBotSettings()
# Your API Key Here: 👇
openai.api_key = ChatBotSettings().OPENAI_API_KEY()
# Your Image Prompt Here: 👇
prompt = "Create a chiropractor back doctor that's a bot working with a human patient"
response = openai.Image.create(
    prompt=prompt,
    n=1,
    size="256x256",
)
print(response["data"][0]["url"])