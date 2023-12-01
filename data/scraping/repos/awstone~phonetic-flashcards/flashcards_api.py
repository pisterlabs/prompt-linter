from fastapi import FastAPI, File, UploadFile, Form, Path, Query
import base64
from fastapi.responses import FileResponse
from PIL import Image
import io
import pathlib
import numpy as np
import cv2
import random
from hf_hub_ctranslate2 import TranslatorCT2fromHfHub, GeneratorCT2fromHfHub
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from diffusers import DiffusionPipeline
import math
from PIL import Image, ImageFont, ImageDraw
import re
import uuid
import json
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response

import openai
import os
from dotenv import load_dotenv
load_dotenv('/home/cendue/awstone/.bashrc')
openai.api_key = os.environ["OPENAI_API_KEY"]


# load both base & refiner
def load_diffusion_model():
    base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
            ).to("cuda")
    refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            ).to("cuda")
    return base, refiner


def call_model(llm_json, base, refiner):
    # objects = [llm_json['pair']['item1']['word'],
    #            llm_json['pair']['item2']['word']]
    visual_prompts = [llm_json['pair']['item1']['visual_prompt'],
                      llm_json['pair']['item2']['visual_prompt']]
    # print("Objects are: ", objects)
    # art_style='cartoon'
    images = []
    for i, prompt in enumerate(visual_prompts):
        # prompt = f"a {art_style} of a {obj}"
        # Define how many steps and what % of steps to be run on each experts (80/20) here
        n_steps = 40
        high_noise_frac = 0.8

        # run both experts
        image = base(
            prompt=prompt,
            num_inference_steps=n_steps,
            # denoising_end=high_noise_frac,
            output_type="latent",
        ).images
        image = refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            # denoising_start=high_noise_frac,
            image=image,
        ).images[0]
        images.append(image)
        # image.save(f"/home/awstone/phonetic-flashcards/images/{obj}.png")
    return images


def generate_images(llm_json, base, refiner):
    images = call_model(llm_json, base, refiner)
    return images

def run_llm(input_string):
    with open('/home/cendue/awstone/phonetic-flashcards/prompts/system-prompt.txt', 'r') as file:
        system_prompt = file.read()
    with open('/home/cendue/awstone/phonetic-flashcards/prompts/unified-prompt.txt', 'r') as file:
        unified_prompt = file.read()
    # append the user input to the unified prompt
    unified_prompt += input_string
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": unified_prompt}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
    )

    return response['choices'][0]['message']['content']
    
# Function to convert PIL image to Base64 string
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")  # Change PNG to JPEG if you prefer
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

app = FastAPI()
base, refiner = load_diffusion_model()

MAX_LENGTH = 1024
origins = [
    "https://vahgarsi.github.io",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
async def root():

    return {"message": "Hello, world!"}

@app.get("/generate/{input_string:path}")
async def generate(response: Response, input_string: str = Path(...)):
    
    response.headers["Access-Control-Allow-Origin"] = "*"

    if len(input_string) > MAX_LENGTH:
        return {"error": "Path too long"}
    print(f'input string: {input_string}')
    llm_json = run_llm(input_string)
    llm_json = json.loads(llm_json)

    print(f'llm output: \n\n {llm_json}')
    
    image_list = generate_images(llm_json, base, refiner)

    # encode the images
    b64_images = []
    for image in image_list:
        image = image.resize((512, 512))
        b64_image = image_to_base64(image)
        b64_images.append(b64_image)
    item1 = llm_json['pair']['item1']
    item2 = llm_json['pair']['item2']
    unique_id = str(uuid.uuid4())
    word1 = item1['word']
    word2 = item2['word']
    visual_prompt1 = item1['visual_prompt']
    visual_prompt2 = item2['visual_prompt']
    sound1 = item1['sound'].replace('/', '')
    sound2 = item2['sound'].replace('/', '')
    ipa1 = item1['ipa'].replace('/', '')
    ipa2 = item2['ipa'].replace('/', '')
    place1 = item1['place']
    place2 = item2['place']
    manner1 = item1['manner']
    manner2 = item2['manner']
    voice1 = item1['voicing']
    voice2 = item2['voicing']
    explanation = llm_json['pair']['explanation']

    # parse input_string for contrast and location
    contrast = 'maximal' # this value is fixed
    if 'initial' in input_string:
        location = 'initial'
    elif 'final' in input_string:
        location = 'final'
    else:
        location = 'No location specified'
    


    # Fake image (you should generate or load an actual image)
    # fake_image_bytes = output
    # output_image_base64 = base64.b64encode(fake_image_bytes).decode("utf-8")

    return {
        'uuid': unique_id,
        'prompt': input_string,
        'contrast': contrast,
        'location': location,
        'sound': sound1,
        'pair':{
            'item1':{
                'word': word1,
                'visual_prompt': visual_prompt1,
                'sound': sound1,
                'ipa': ipa1,
                'place': place1,
                'manner': manner1,
                'voicing': voice1,
                'b64_image': b64_images[0]
                },
            'item2':{
                'word': word2,
                'visual_prompt': visual_prompt2,
                'sound': sound2,
                'ipa': ipa2,
                'place': place2,
                'manner': manner2,
                'voicing': voice2,
                'b64_image': b64_images[1]
                }
            },
        'explanation': explanation
        }
