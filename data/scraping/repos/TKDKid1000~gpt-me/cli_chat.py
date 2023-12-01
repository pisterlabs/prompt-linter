import json
import os
import pickle

import openai
from dotenv import load_dotenv

from gptme.assistant import Assistant
from gptme.text_styler import TextStyler

load_dotenv(dotenv_path=".env")

openai.api_key = os.environ["OPENAI_KEY"]

with open("personality.txt", encoding="utf8") as personality_file:
    personality = personality_file.read()

with open(os.environ["EMBEDDINGS_FILE"], "rb") as embeddings_file:
    embeddings, transcript = pickle.load(embeddings_file)

with open(os.environ["STYLER_FILE"], "r") as style_file:
    style_json = json.load(style_file)
    text_styler = TextStyler()
    text_styler.style = style_json["style"]

assistant = Assistant(
    embeddings=embeddings,
    memories=transcript,
    personality=personality,
    name=os.environ["ASSISTANT_NAME"],
    text_styler=text_styler
)

while True:
    message = input("user: ")

    images = input("images: ").split(" ")

    print(images)

    assistant.send_message(text=message, images=images if images[0] != "" else [])

    print(f"assistant: {assistant.conversation.messages[-1].content}")
