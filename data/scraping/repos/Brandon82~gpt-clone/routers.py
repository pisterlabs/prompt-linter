from fastapi import APIRouter, HTTPException
from models.chat_input import ChatInput
from models.chat_response import ChatResponse
from models.image_input import ImageInput
from models.image_response import ImageResponse
import openai
import json
import os

router = APIRouter()

file_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
print(f"Attempting to open config file at: {file_path}")  # Debug line

with open(file_path) as f:
    config = json.load(f)

api_key = config['keys']['openAI']

@router.post("/chat/", response_model=ChatResponse)
async def chat(chat_input: ChatInput):
    try:
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model=chat_input.engine,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": chat_input.prompt
                }
            ],
            max_tokens=chat_input.max_tokens
        )
        message = response['choices'][0]['message']['content']
        print(f"Chat request: {chat_input.dict()}, Response: {message}")
        return {"message": message}
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/image/", response_model=ImageResponse)
async def generate_image(image_input: ImageInput):
    try:
        openai.api_key = api_key
        response = openai.Image.create(
            prompt=image_input.prompt,
            n=image_input.num_images,
            size=image_input.image_size
        )
        image_url = response['data'][0]['url']
        print(f"Image request: {image_input.dict()}, Response URL: {image_url}")
        return {"url": image_url}
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
