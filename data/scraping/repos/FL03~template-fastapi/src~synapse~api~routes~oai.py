"""
    Appellation: openai
    Contrib: FL03 <jo3mccain@icloud.com>
    Description: ... Summary ...
"""
from fastapi import APIRouter, Form, HTTPException
from typing import List, Dict

from synapse.data.messages import Message, Status
from synapse.data.models.users import User
import openai, os

openai.api_key = os.getenv("OPENAI_API_KEY")
router = APIRouter(prefix="/openai", tags=["OpenAI"])

def clean_completion(completion: openai.Completion): 
    tmp = [i.text.strip("?\n\n") for i in completion.choices]
    if len(tmp) > 1:
        return tmp
    else:
        return tmp[0]

def create_completion(model: str, prompt: str, **kwargs):
    return openai.Completion.create(model=model, prompt=prompt, **kwargs)

@router.get("/", response_model=Dict)
async def landing():
    return Message(message="OpenAI Endpoints")


@router.post("/chatgpt")
async def chatgpt3_completion(prompt: str = Form(), temp: float = 0.5, max_tokens: int = 2000):
    completion = create_completion("text-davinci-003", prompt, temperature=temp, max_tokens=max_tokens)
    return Message(message=clean_completion(completion))


@router.post("/codex")
async def codex_completion(prompt: str = Form(), temp: float = 0.5, max_tokens: int = 2000):
    completion = create_completion("text-davinci-002", prompt, temperature=temp, max_tokens=max_tokens)
    return clean_completion(completion)
