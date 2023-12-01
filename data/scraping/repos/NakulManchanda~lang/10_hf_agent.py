## Google Colab from Hugging face
## https://colab.research.google.com/drive/1c7MHD-T1forUPGcC_jlwsIptOzpG3hSj
## Blog tutorial
## https://towardsdatascience.com/hugging-face-transformers-agent-3a01cf3669ac
## Can we use LLMs to decide which model to use, write code, run code, and generate results?

import logging
import os
import sys
from dotenv import load_dotenv
load_dotenv()

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


## KEYS
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
SERPAPI_API_KEY=os.environ["SERPAPI_API_KEY"]
HF_TOKEN=os.environ["HF_TOKEN"]

## Config
PDF_ROOT_DIR=os.environ["PDF_ROOT_DIR"]
OUTPUT_DIR=os.environ["OUTPUT_DIR"]

## utils
from playsound import playsound
import soundfile as sf
def play_audio(audio, wav_file_name):
    sf.write(f"{OUTPUT_DIR}/{wav_file_name}.wav", audio.numpy(), samplerate=16000)
    playsound(f"{OUTPUT_DIR}/{wav_file_name}.wav")


## MAIN

# login to huggingface
from huggingface_hub import login
login(HF_TOKEN)

# initialize openai agent from huggingface
from transformers.tools import OpenAiAgent
agent = OpenAiAgent(model="text-davinci-003", api_key=OPENAI_API_KEY)
print("OpenAI is initialized 💪")

## EXAMPLE 1: Generate image from text
def example1():
    # generate text to image from huggingface agent - it will choose right tool for you
    # `image_generator` to generate an image according to the prompt.
    boat = agent.run("Generate an image of a boat in the water")
    # print(boat)

    # save image to disk
    from PIL import Image
    boat.save(f'{OUTPUT_DIR}/boat.png')
    return boat

## EXAMPLE 2: Generate text based on image
def example2():
    # `image_captioner` to generate a caption for the image.
    boat = example1()
    caption = agent.run("Can you caption the `boat_image`?", boat_image=boat)
    print(f'CAPTION: {caption}')

## EXAMPLE 3: Generate image based on text, generate caption based on image, and generate audio based on caption
def example3():
    # `image_generator` to generate an image of a boat, then `text_reader` to read out loud the contents of the image.
    # Agents vary in competency and their capacity to handle several instructions at once; 
    # however the strongest of them (such as OpenAI's) are able to handle complex instructions 
    # such as the following three-part instruction
    audio = agent.run("Can you generate an image of a boat? Please read out loud the contents of the image afterwards")
    play_audio(audio, 'boat_image_desc_audio')

example2()