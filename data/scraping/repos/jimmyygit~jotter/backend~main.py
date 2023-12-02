from fastapi import FastAPI
import cohere
from typing import List
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()


app = FastAPI()
co = cohere.Client(os.getenv("secret_key"))

origins = [
    "http://localhost:5173",
    "http://localhost:4173",
    "https://jotter-y4es.onrender.com",
    "https://jotters.netlify.app",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/cohere")
async def cohere_chat():
    response = co.generate(
        prompt="Please explain to me how LLMs work",
    )
    return {"message": response}


class Suggestion(BaseModel):
    note: str


class Category(BaseModel):
    categories: List[str]
    note: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "categories": ["Family", "School", "Career", "To-do"],
                    "note": "Apply to Google",
                }
            ]
        }
    }


@app.post("/categorize")
async def categorize(category: Category):
    BASE_PROMPT = f"""We have the following categories: {", ".join(category.categories)}. Which of these categories would you classify the following message under: {category.note}."""

    response = co.generate(model="command", prompt=BASE_PROMPT)
    return_category = response.generations[0].text.strip()
    return {"category": return_category}


@app.post("/suggestion")
async def suggestion(prompt: Suggestion):
    SUGG_PROMPT = f""" Based on this note: {prompt.note}, provide 1 very brief and very funny suggestion on how to accomplish the aforementioned message.
    Make sure that suggestion is VERY funny, but not mean to anyone--they should be harmless jokes and not contain any violence. The suggestion must just be one short sentence."""
    suggestion = co.generate(max_tokens = 100, model="command", prompt=SUGG_PROMPT)
    return_suggestion = suggestion.generations[0].text.strip()
    return {"suggestion": return_suggestion}
