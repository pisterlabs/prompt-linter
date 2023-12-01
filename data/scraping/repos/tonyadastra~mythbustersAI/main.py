from dotenv import load_dotenv
from fastapi import APIRouter
import anthropic
import os
import xml.etree.ElementTree as ET
import json
import random
import sys
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import io
from backend.fact_checker import FactChecker
from backend.claim_extractor import ClaimExtractor
import base64


from typing import Optional

load_dotenv()

def init_client():
    client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return client

client = init_client()

app = APIRouter()

class QuestionReq(BaseModel):
    question: Optional[str] = ""

class ClaimInput(BaseModel):
    claim: str
    speaker: str
    opponent: str

class ParagraphInput(BaseModel):
    paragraph: str
    speaker: str
    opponent: str
    moderator: str

@app.post("/generate")
async def generate(req: QuestionReq):
    print("received question: ", req)
    response = generate_moderator_questions(client, req.question) 
    return response

@app.post("/fact_check")
def fact_check(input: ClaimInput):
    claim = dict()
    claim["claim"]=input.claim
    claim["speaker"] = input.speaker
    claim["opponent"] = input.opponent

    truthGPT = FactChecker(client)
    fact_checking_result = truthGPT.factCheck(claim)
    print(fact_checking_result)
    return fact_checking_result

@app.post("/extract_claims")
def extract_claims(input: ParagraphInput):
    paragraph = dict()
    paragraph["paragraph"]=input.paragraph
    paragraph["speaker"] = input.speaker
    paragraph["opponent"] = input.opponent
    paragraph["moderator"] = input.moderator

    extractor = ClaimExtractor(client)
    claims = extractor.extractClaims(paragraph)
    print(claims)
    return claims
    
def generate_moderator_questions(client, question = None):

    if question:
        print("Input question is not empty")
        random_question = question

    else:
        print("Script called without args")
        with open(r"./backend/prompts/questions.txt") as f:
            questions = f.read().split('\n')
            questions = list(filter(None, questions)) 
            random_question = random.choice(questions)
    
    print(random_question)

    with open(r"./backend/prompts/flow.md") as f:
        prompt = f.read()
        updated_prompt = prompt.replace("$question", random_question)
    
    resp = client.completions.create(
        model="claude-instant-1",
        # model="claude-2",
        prompt=f"""\n\nHuman: {updated_prompt}
        \n\nAssistant: Here is the response in XML format:\n\n""",
        max_tokens_to_sample=1000,
        stream=False,
        )
    xml_string = resp.completion
    print(xml_string)

    root = ET.fromstring(xml_string)

    json_data = {}

    json_data['elon_musk_question'] = root.find('question').text   
    json_data['joe_biden_answer'] = root.find('answer1').text
    json_data['donald_trump_answer'] = root.find('answer2').text
    json_data['joe_biden_rebuttal'] = root.find('rebuttal1').text
    json_data['donald_trump_rebuttal'] = root.find('rebuttal2').text

    return json_data


class TextToSpeechReq(BaseModel):
    role: str
    transcript: str
    stream: bool

@app.post("/text-to-speech")
async def generate(req: TextToSpeechReq):
    apikey = os.getenv("xi-api-key")

    response = generate_audio_stream(apikey, req.role, req.transcript, req.stream)
    return response

def generate_audio_stream(apikey, role, transcript, stream):
    elon_video_id = "t7EoxTx22W0catHmnoak"
    if role == "elon":
        video_id = elon_video_id
    elif role == "biden":
        video_id = "rw1hja2xibJD2qJzpBHJ"
    elif role == "trump":
        video_id = "aGHgh4gMY0lKysbmJkyc"
    else:
        video_id = elon_video_id

    CHUNK_SIZE = 1024

    if stream:
        url = "https://api.elevenlabs.io/v1/text-to-speech/{video_id}/stream"
    else:
        url = "https://api.elevenlabs.io/v1/text-to-speech/{video_id}"
    url = url.format(video_id=video_id)

    print(url)

    headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": apikey
    }

    data = {
    "text": transcript,
    "model_id": "eleven_monolingual_v1",
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.5
    }
    }

    response = requests.post(url, json=data, headers=headers, stream=True)
    print(response.status_code)

    audio_stream = io.BytesIO()

    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
        if chunk:
            audio_stream.write(chunk)

    audio_stream.seek(0)

    if stream:
        return audio_stream
    else:
        audio_data_bytes = audio_stream.read()
        audio_stream.close()
        audio_data_base64 = base64.b64encode(audio_data_bytes).decode('utf-8')
        audio_data_json = {
            'audio_bytes': audio_data_base64
        }
        return audio_data_json