from http.client import HTTPException
from os import environ
from fastapi import FastAPI, Query
from typing import List
from fastapi.staticfiles import StaticFiles
import dotenv
import cohere 
import os
from pydantic import BaseModel

dotenv.load_dotenv()

app = FastAPI()


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as ex:
            if ex.status_code == 404:
                return await super().get_response("index.html", scope)
            else:
                raise ex



co = cohere.Client(os.getenv('COHERE_API_KEY')) 

class Recipe(BaseModel):
    ingredients: List[str]

@app.post("/create_recipe")
def create_recipe(recipe: Recipe):
    ingredients = recipe.ingredients
    response = co.generate( 
        model='699a8d5b-cd69-4802-b81a-80ad2fd8eaa3-ft',#model='5ac071ae-9dee-4d7e-932c-1a20d7df4483-ft', 
        prompt = "Ingredients: " + "\n".join(ingredients) + "\nTitle: ", 
        max_tokens=200, 
        temperature=0.9, 
        k=0, 
        p=0.75, 
        frequency_penalty=0.1, 
        presence_penalty=0, 
        stop_sequences=["$SEP$", "$END$"], 
        return_likelihoods='NONE')
    text = response.generations[0].text
    lines = text.split('\n')
    title = lines[0]
    recipe = '\n'.join(lines[1:])
    
    return {'title': title, 'recipe': recipe, 'full': text}

app.mount("/", SPAStaticFiles(directory="./website/build", html=True), name="app")