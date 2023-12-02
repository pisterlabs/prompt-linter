
import openai
import json
from app import app, db
import urllib.request
import os
import requests
import re
from langchain.document_loaders import WebBaseLoader
from app.models import Recipe, Ingredients, Naehrwerte, Instructions
from werkzeug.utils import secure_filename

from app.ai_config import *

#importing modules
import pytesseract

#importing opencv
import cv2

def image_to_text(path):
    img = cv2.imread(path)
    #Resize the image
    img = cv2.resize(img, None, fx=4, fy=4,  interpolation=cv2.INTER_CUBIC)
    #Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang='deu',config="--psm 1")
    return text

def save_image(uploaded_file):
    filename = secure_filename(uploaded_file.filename)
    path = app.config["UPLOAD_FOLDER"]+filename
    uploaded_file.save(path)
    return path


# define the function
def get_recipe(dish_type,ingredients,recipe_type):
    if recipe_type  == 'y':
        recipe_type = "vegetarisches"
    else:
        recipe_type = ''

    query = f"Erstelle ein {recipe_type} {dish_type} Rezept mit folgenden Zutaten: {ingredients}"
    response = retrieval_qa(query)
    
    return response["result"].dict()


#create an image using DALL-E
def get_image(title):
    PROMPT = f"{title}"
    print(PROMPT)
    response = openai.Image.create(
        model="dall-e-3",
        prompt=PROMPT,
        n=1,
        size="1024x1024",
        quality="standard"
    )   
    
    if "data" in response:
        for key, obj in enumerate(response["data"]):
            id = obj['url'].split("img-")[1].split(".png")[0]
            filename = os.path.join(app.root_path, 'static', 'img', f'{id}_recipe_image.png')
            print(filename)
            urllib.request.urlretrieve(obj['url'], filename)
        print('Images have been downloaded and saved locally')
    else:
        print("Failed to generate image")
    return f'{str(id)}_recipe_image.png'


#function that uses requets lib to extract html of a web page and prettyfies it with bs4
def extract_from_website(url):
    loader = WebBaseLoader(url)
    body = loader.scrape().find('body').text
    return re.sub(r'\s+', " ", body)

def save_recipe(recipe_data, dish_type):
    print("asking DALL_E for help ...")
    print(recipe_data['beschreibung'])
    image = get_image(recipe_data['prompt'])
    print('storing info in database ...')

    #create a new recipe
    recipe = Recipe(title=recipe_data['title'],
                    prompt = recipe_data['prompt'],
                    beschreibung = recipe_data['beschreibung'],
                    portionen = recipe_data['portionen'],
                    recipe_type = recipe_data['recipe_type'], 
                    tipp =  recipe_data['tipp'],
                    image_id=image,
                    dish_type = dish_type,
                    ingredients = [Ingredients(
                        name = ingredient['name'],
                        unit=ingredient['unit'],
                        amount=ingredient['amount']
                        ) for ingredient in recipe_data['ingredients']],
                    naehrwerte = [Naehrwerte(
                        name = naehrwert['name'],
                        unit=naehrwert['unit'],
                        amount=naehrwert['amount']
                        ) for naehrwert in recipe_data['naehrwerte']],
                    instructions = [Instructions(
                        instruction=instruction
                        ) for instruction in recipe_data['instructions']]
                    )
    db.session.add(recipe)
    db.session.commit()
    return(recipe.id)