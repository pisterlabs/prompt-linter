from fastapi import APIRouter, UploadFile, Form
import requests
from typing import Annotated
from diffusers.utils import load_image
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from ...config import BaseData, control_net_pipe, NUM_INFERENCE_STEPS, ImageReq
from ..data.data_model import User
from ..data.db import Data
import os
import base64
import openai
import readline
import re

router = APIRouter()

storage = Data.get_storage_instance()

bucket = storage.bucket()

def has_voice(username: str, headers, delete: bool=False):
    voices_url = f'https://api.elevenlabs.io/v1/voices'
    voices_response = requests.get(voices_url, headers=headers)

    # This snippet checks to see if this person already has a voice and removes
    # that old voice upon recording their new voice.

    if voices_response.ok:
        for voice in voices_response.json()["voices"]:
            if voice["name"] == f'{username} voice':
                print("FOUND VOICE")
                if delete:
                    delete_voice_url = f'https://api.elevenlabs.io/v1/voices/{voice["voice_id"]}'
                    requests.delete(delete_voice_url, headers=headers)
                return True
    return False


@router.get("/health")
def root():
    return {"status": "OK"}

@router.get("/has_prompt/{username}")
def has_prompt(username: str):
    if username != "":
        file_path = f'audio/{username}-ai-voice.mp3'
        blob = bucket.blob(file_path)

        if blob.exists():
            return {"success": True}
        return {"success": False}

@router.post("/get_orig_portrait")
def get_orig_portrait(user: BaseData):

    if user.username != "":
        file_path = f'images/{user.username}-orig-portrait.jpg'
        blob = bucket.blob(file_path)

        # Download the file as bytes
        if blob.exists():
            file_bytes = blob.download_as_bytes()

            return base64.b64encode(file_bytes)
    
@router.post("/get_ai_portrait")
def get_orig_portrait(user: BaseData):
    if user.username != "":
        file_path = f'images/{user.username}-ai-portrait.jpg'
        blob = bucket.blob(file_path)

        # Download the file as bytes
        if blob.exists():
            file_bytes = blob.download_as_bytes()

            return base64.b64encode(file_bytes)


@router.post("/signup")
def login(user: BaseData):
    if user.username == "":
        return {"success": False, "message": "Error: Username field was empty"}
    
    db = Data.get_db_instance()

    users_ref = db.reference('/users')

    users = users_ref.get()

    for usr in users.items():
        if usr[1]["user_id"] == user.username:
            return { "success": False, "message": "User Already exists, try logging in instead"}
    
    new_user = User(user.username)
    new_user.initialize()
    new_user.save()

    return {"success": True}


@router.post("/login")
def login(user: BaseData):
    if user.username == "":
        return {"success": False, "message": "Error: Username field was empty"}
    
    db = Data.get_db_instance()

    users_ref = db.reference('/users')

    users = users_ref.get()

    for usr in users.items():
        if usr[1]["user_id"] == user.username:
            return { "success": True }
    
    return { "success": False, "message": "It looks like you don't have an account yet, go ahead and sign up"}

    

@router.get("/has_voice/{username}")
def check_has_recording(username: str):

    headers = {
        "Accept": "application/json",
        "xi-api-key": os.environ["VOICE_API_KEY"]
    }

    return has_voice(username, headers)

@router.post("/generate_canny")
async def generate_canny(username: Annotated[str, Form()], prompt: Annotated[str, Form()], image: Annotated[UploadFile, Form()]):

    image_data = image.file.read()

    image_stream = BytesIO(image_data)

    image = Image.open(image_stream)

    negative_prompt = 'low quality, bad quality, sketches'
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    image = control_net_pipe(prompt, image, num_inference_steps=NUM_INFERENCE_STEPS, negative_prompt=negative_prompt).images[0]
    
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")  # You can use JPEG or other formats as needed
    image_bytes = image_bytes.getvalue()

    canny_path = f"images/{username}-canny.jpg"
    canny_blob = bucket.blob(canny_path)

    canny_blob.upload_from_string(image_bytes, content_type="image/jpeg")

    base64_image = base64.b64encode(image_bytes).decode()


    return {"image_base64": base64_image}

@router.get("/get_canny/{username}")
def get_canny(username: str):
    if username != "":

        file_path = f'images/{username}-canny.jpg'
        blob = bucket.blob(file_path)

        if blob.exists():
            file_bytes = blob.download_as_bytes()

            return base64.b64encode(file_bytes)


@router.get("/love_letter/{username}")
def get_love_letter(username: str):
    user = User(username)

    return user.get("love_letter", None)


@router.post("/love_letter")
def generate_love_letter(req: ImageReq):
    username = req.username
    prompt = req.prompt

    user = User(username)

    response = do_openai_query( prompt=prompt )
    user["love_letter"] = response

    user.save()

    return response



def do_openai_query( prompt, max_tokens=2048, temperature=1.0 ):


    model = 'gpt-3.5-turbo'

    messages = {"role": "system", "content": prompt },


    response = openai.ChatCompletion.create(

        messages=messages,

        model=model,

        max_tokens=max_tokens,

        temperature=temperature,

        )


    return response['choices'][0]['message']['content']



@router.get("/high_score/{username}")
def get_score(username: str):
    user = User(username)

    return user.get("high_score", None)

@router.post("/score_pitch")
def score_pitch(pitch: ImageReq):
    my_pitch = pitch.prompt
    username = pitch.username

    user = User(username)

    PROMPT = f"""The following paragraph is a sales pitch. The salesman is trying to sell blue jerseys to the University of Utah football team; this is challenging, because the University of Utah has a strong rivalry with BYU, whose primary colors are blue and white. However, the University is abandoning its long-standing color of red in favor of some sort of blue.



    Analyze the sales pitch, and judge how good it is:



    Beginning of pitch:

    {my_pitch}

    End of pitch



    Now, calculate how much of your budget you're willing to spend based on the quality of the pitch. Your budget is $100M. If the pitch is absolutely perfect, you will spend all $100M. If the pitch was awful, you will spend $0M.



    Answer the following question with only a number. Do NOT include any additional text.



    Based on the quality of the pitch, you will spend $"""

    response = do_openai_query( PROMPT )

    match = re.match(r'^\d+', response)
    numeric_values = match.group()

    if user.get("high_score", None) and user["high_score"] < int(numeric_values):
        user["high_score"] = int(numeric_values)
    elif not user.get("high_score", None):
        user["high_score"] = int(numeric_values)
    
    user.save()

    return numeric_values


@router.post("/generate_pitch")
def generate_pitch(user_prompt: ImageReq):

    response = do_openai_query(user_prompt.prompt)

    user = User(user_prompt.username)
    user["gpt_prompt"] = response
    user.save()

    return response

@router.get("/get_pitch/{username}")
def generate_pitch(username: str):


    user = User(username)


    return user.get("gpt_prompt", None)


@router.post("/text_to_image")
def text_to_image(image_req: ImageReq):
    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    }

    data = {
        "prompt": image_req.prompt,
        "n": 1,
        "size": "1024x1024"
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        result = response.json()
        return result['data'][0]['url']

