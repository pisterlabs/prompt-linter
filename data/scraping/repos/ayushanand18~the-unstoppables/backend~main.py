"""Backend API using FastAPI"""

import json
import os
import requests
import sqlite3
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import firestore, initialize_app, credentials
from pydantic import BaseModel
from transformers import DiffusionImageGenerator
import cohere
import spacy

#---------------------
# Loading and setting
# essential stuff
#---------------------
load_dotenv()
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class GenerateImageRequest(BaseModel):
    prompt: str
    num_images: int = 1

class GenerateImageResponse(BaseModel):
    images: list[str]

class TokenizeRequest:
    def __init__(self, prompt: str):
        self.prompt = prompt

class TokenizeResponse:
    def __init__(self, important_words: list[str]):
        self.important_words = important_words

class ScrapedImagesResponse:
    def __init__(self, images: list[str]):
        self.images = images

#---------------------
# Environment
#---------------------
COHERE_KEY = os.environ['COHERE_KEY']
# cred = credentials.Certificate("ddf75-e3484fe17c32.json")
# initialize_app(cred)
# db = firestore.client()
co = cohere.Client(COHERE_KEY)
model = DiffusionImageGenerator.from_pretrained("openai/diffusion:main")
nlp = spacy.load("en_core_web_sm")
db_connection = sqlite3.connect("db.qlite3")
db_cursor = db_connection.cursor()

#----------------------
# API Endpoints
#----------------------
@app.get("/")
async def root():
    """Main Root Function"""
    return {"message": "Hello World"}


@app.post("/parse_feedback")
async def parse_feedback(request: Request):
    """Parse User Feedback into a JSON based output"""
    try:

        req_body = await request.body()
        req_data = json.loads(req_body)
        prompt = req_data['prompt']

        message = "you are an expert parser. now wear your hat of a parser and parse the sentence I am going to give you. for example, for the sentence 'this the most terrible product I have seen, change the color of the skirt and then show me'. the output should be a JSON as \{'type': False, 'element':'pant'\} and for 'this dress is great. I like this but change the sandals. the output should be \{'type': False, 'element':'shoe'\}. and for 'great suggestion. I like this the output should be \{'type': True, 'element':''\}. Now generate for the sentence: "+prompt+" DO NOT WRITE ANYTHING OTHER THAN THE JSON OUTPUT."  
        response = co.generate(
            prompt = message,
        )

        return response[0]
    except BaseException as error:
        return {
            "status": "error",
            "message": str(error),
        }

@app.post("send-message")
async def send_message(request: Request):
    try:
        req_body = await request.body()
        req_data = json.loads(req_body)
        prompt = req_data['prompt']

        message = f"you are an expert fashion generator bot. answer this question now. ${prompt}"
        response = co.generate(
            prompt = message,
        )

        return response[0]
    except BaseException as error:
        return {
            "status": "error",
            "message": str(error),
        }

def tokenize_prompt(request: TokenizeRequest):
    doc = nlp(request.prompt)
    
    # Extract important words based on POS (Part-of-Speech) tags
    important_words = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
    
    return TokenizeResponse(important_words=important_words)

@app.post("/generate-image", response_model=GenerateImageResponse)
async def generate_image(request: GenerateImageRequest):
    """Generate Images based on Fashion prompt"""
    try:
        prompt = tokenize_prompt(request.prompt)
        images = model.generate_images(prompt, num_images=4)
        return {"images": images}
    except Exception as error:
        return {
            "status": "error",
            "message": str(error),
        }

def get_fashion_tags(image_url):
    """Get Fashion Tags from locally hosted ML model"""
    fashion_tagger_api_url = 'http://localhost:9000'
    
    response = requests.post(fashion_tagger_api_url, json={"image_url": image_url}, timeout=3000)
    if response.status_code == 200:
        tags = response.json().get("tags", [])
        return tags
    else:
        return []

@app.get("/scrape-instagram-with-tags", response_model=ScrapedImagesResponse)
async def scrape_instagram_with_tags():
    """Scrape Instagram Fashion to get recent trending fashion"""
    base_url = "https://www.instagram.com/explore/tags/fashion/"
    
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            image_tags = soup.find_all("img")
            
            images_with_tags = []
            for img in image_tags:
                if img.has_attr("src"):
                    image_url = img["src"]
                    age, gender, region, tags = get_fashion_tags(image_url)
                    images_with_tags.append({"image_url": image_url, "age": age, "gender": gender, "region": region, "tags": tags})
            
            return ScrapedImagesResponse(images=images_with_tags)
        else:
            return ScrapedImagesResponse(images=[])
    except Exception as e:
        return ScrapedImagesResponse(images=[])

def create_table():
    """Create the 'fashion_tags' table if it doesn't exist"""
    db_cursor.execute("""
    CREATE TABLE IF NOT EXISTS fashion_tags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_url TEXT,
        age INTEGER,
        region TEXT,
        gender BOOLEAN NOT NULL,
        tag TEXT
    )
    """)
    db_connection.commit()

def insert_tags(image_url, age, region, gender, tags):
    """Insert image URL, tags, etc into the database"""
    for tag in tags:
        db_cursor.execute("INSERT INTO fashion_tags (image_url, age, region, gender, tag) VALUES (?, ?)", (image_url, age, region, gender, tag))
    db_connection.commit()