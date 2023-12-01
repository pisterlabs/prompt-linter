from typing import List
import requests
import cohere  
import os
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
EDAMAM_API_KEY = os.getenv("EDAMAM_API_KEY")
EDAMAM_API_URL = "https://api.edamam.com/api/recipes/v2"
EDAMAM_APP_ID = os.getenv("EDAMAM_APP_ID")

def generate_recipe(food_name: str, ingredients: List[str]) -> str:
    if COHERE_API_KEY == None:
        raise Exception("API key not found.")
    co = cohere.Client(COHERE_API_KEY)
    prompt = f"Give me a recipe in JSON for {food_name} that uses the following ingredients: "
    for ingredient in ingredients:
        prompt +=  "\n " + ingredient
    response = co.generate(  
        model='command-nightly',  
        prompt = prompt,  
        max_tokens=200,  
        temperature=0.750)

    if response.generations == None:
        raise Exception("No response from API.")

    return response.generations[0].text
def generate_llm_recipes(ingredients: List[str]) -> str:
    if COHERE_API_KEY == None:
        raise Exception("API key not found.")
    co = cohere.Client(COHERE_API_KEY)
    prompt = "Give me a list of recipes (maximum 3) with steps in JSON format that use the following ingredients: "

    for ingredient in ingredients:
        prompt +=  "\n " + ingredient
    prompt += "\n Give a JSON format of an array with objects with property keys \"name\", \"ingredients\", \"steps\". Keep your answer relatively short. Separate the steps into individual strings in their respective arrays and include commas for each element. Make sure you don't leave trailing commas for the end of arrays. " 
    response = co.generate(  
        model='command-nightly',  
        prompt = prompt,  
        max_tokens=2000,  
        temperature=0.750)

    if response.generations == None:
        raise Exception("No response from API.")

    print("".join([elem.text for elem in response.generations]))
    return response.generations[0].text

def get_edamam_recipe(ingredients: List[str]) -> str:
    if EDAMAM_API_KEY == None or EDAMAM_APP_ID == None:
        raise Exception("oh no")
    query_str = f"?app_id=98d69878&app_key={EDAMAM_API_KEY}"
    query_str += "&q=" + '+'.join(ingredients)
    print(query_str)
    r = requests.get(f"{EDAMAM_API_URL}{query_str}", 
                     params={"app_key": EDAMAM_API_KEY, 
                             "app_id": EDAMAM_APP_ID, 
                             "ingredients": ingredients,
                             "type": "public",
                             }
                     )
    recipes = r.json()["hits"]
    recipes = [{ "title": x["recipe"]["label"], "ingredients": [ y["text"] for y in x["recipe"]["ingredients"]] } for x in recipes]

    return str(recipes)

if __name__ == "__main__":
    ingredients = ["ham", "rice", "chicken", "teriyaki"]
    #get_edamam_recipe(ingredients)
