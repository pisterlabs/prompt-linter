from fastapi import FastAPI
import json
from openai_api import get_chat_gpt_output

app = FastAPI()

@app.get("/")
async def root():
    with open("example/profile.json") as fp:
        profile = json.load(fp)
    return profile


@app.get("/plaintext/")
async def root():
    with open("example/profile.json") as fp:
        profile = json.load(fp)
    text = get_chat_gpt_output(f"Given the following json description of a person, generate a biography of them including all the information. {str(profile)}")
    return text

