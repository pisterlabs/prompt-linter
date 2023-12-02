import streamlit as st
import json
import pandas as pd
import openai
from dotenv import load_dotenv
import os

load_dotenv()


st.set_page_config(page_title="Saved Recipes", page_icon="📖")

st.title("📖 Saved Recipes")


with open("./assets/recipes.txt", "r") as f:
    data = f.read()

# data = json.loads(data)

data = data.replace("'", "\"").strip()
list_of_json = data.split('\n\n')

col1, col2 = st.columns([1, 2])

col1.subheader('Recipe List')


recipes = []

for j in list_of_json:
    d = json.loads(j)
    recipes.append(d['recipe_name'])

which_recipe = col1.radio(
    "Pick any of the recipes you saved",
    recipes)


col2.subheader(which_recipe)

# prompt = ""

# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


# response = openai.Image.create(

#     prompt=prompt,

#     n=1,

#     size="512x512",

# )

# st.text(response["data"][0]["url"])

col2.subheader("Ingredients:\n")


ingredients = json.loads(list_of_json[recipes.index(which_recipe)])[
    'ingredients']

ingredients_list = ""
for item in ingredients:
    ingredients_list += f"* {item[0]}: {item[1]} {item[2]}\n"
col2.markdown(f"{ingredients_list}\n")

col2.subheader("Cooking Steps:\n")


steps_pretty = ""
for step in json.loads(list_of_json[recipes.index(which_recipe)])['cooking_steps']:
    steps_pretty += f"- {step}\n"

col2.markdown(f"{steps_pretty}\n")
