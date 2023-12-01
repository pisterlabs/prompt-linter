from dotenv import find_dotenv, load_dotenv
# from transformers import pipeline

import os
import requests

# from apikey import OPENAI_API_KEY, HF_API_TOKEN
from langchain import PromptTemplate, LLMChain, OpenAI

IMG2TEXT_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"

load_dotenv(find_dotenv())

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HG_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# img2text
def img2text(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(IMG2TEXT_API_URL, headers=HG_HEADERS, data=data)

    print(response.json())
    return response.json()

# llm
def generate_story(scenario):
    template = """
    You are a story teller:
    You can generate a short story based on a simple narrative, the story should be no more than 20 words;

    Context: {scenario}
    Story:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["scenario"])

    llm = OpenAI(temperature=0.9)
    story_llm = LLMChain(llm=llm, prompt=prompt, verbose=True, output_key='title')

    story = story_llm.run(scenario=scenario)

    print(story)
    return story

# text to speech
def text_to_speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    payload = {
        "inputs": message
    }

    response = requests.post(API_URL, headers=HG_HEADERS, json=payload)
    # print(response.json())
    with open("audio.flac", 'wb') as file:
        file.write(response.content)

scenario = img2text("Snowboard.jpeg")
story = generate_story(scenario)
text_to_speech(story)
