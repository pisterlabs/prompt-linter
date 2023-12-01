from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
# from langchain import PromptTemplate, LLMChain, OpenAI
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate 

import os, requests

load_dotenv(find_dotenv())

# image to text, url is the link to the image
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    
    text = image_to_text(url)[0]['generated_text']
    print(text)
    return text

#llm - we take the text from the above function and convert it to a story
def generate_story(scenario):
    template = """You are storyteller; you can generate a short story based on a simple narrative, the story should be no more than 20 words;
    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    story_llm = LLMChain(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1), prompt=prompt, verbose=True)
    story = story_llm.predict(scenario=scenario)
    print(story)
    return story

def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"""Bearer {os.getenv("HUGGINGFACE_API")}"""}
    payload = {"inputs": message}
    response = requests.post(API_URL, headers=headers, json=payload)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)

scenario = img2text("fun.jpg")
story=generate_story(scenario)
text2speech(story)
        
    