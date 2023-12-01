import gradio as gr
import os
import io
import numpy as np
from openai_interface import OpenAIInterface
from parse_ingredients import parse_ingredients
from test import workflow
import boto3
from fridge_detector import make_fridge_request
from doordash import order

AWS_ACCESS_KEY = 'AKIAXGZCIJYW3PH2UFNN'
AWS_SECRET_ACCESS_KEY = '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'
AWS_STORAGE_BUCKET_NAME = 'cartesan'

def get_ingredients_recipe(text, fridge_image):
    # get fridge ingredients
    image_url = upload_to_s3(fridge_image)
    fridge_contents = make_fridge_request(image_url)

    # make the call and get the recipe
    openai_interface = OpenAIInterface(api_key='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')
    ingredients_prompt = f"""
    You are helping me make the food I want to eat. I want to cook {text} today.
    Please give me an itemized list of ingredients in numbered list format, where each item is in the format of 'Item:Quantity'. Then, after the heading recipe, give 
    me a step by step recipe.
    """
    numbered_ingredients = openai_interface.predict_text(ingredients_prompt, temp=1)
    numbered_ingredients, recipe = numbered_ingredients.split("Recipe:", 1)
    ingredients = parse_ingredients(numbered_ingredients)
    to_purchase, fridge_contents = workflow(ingredients, existing_ingredients=fridge_contents)

    print('to_purchase', to_purchase)
    print('fridge_contents', fridge_contents)
    print('Recipe', recipe)

    items_only = [tp.split(':')[0] for tp in to_purchase]
    order(items_only)
    return recipe.strip(), fridge_contents, '\n'.join(items_only)

def upload_to_s3(image):
    # save fridge image in memory
    in_mem_file = io.BytesIO()
    image.save(in_mem_file, format = "PNG")
    in_mem_file.seek(0)

    # upload to s3
    object_name = f'fridge_{np.random.randint(0, 1000000)}.png'
    client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    client.upload_fileobj(
        in_mem_file,
        AWS_STORAGE_BUCKET_NAME, # s3 bucket name
        object_name, # s3 bucket key
        ExtraArgs={'ACL':'public-read'}
    )

    return f'https://{AWS_STORAGE_BUCKET_NAME}.s3.us-west-1.amazonaws.com/{object_name}'

def mirror(x):
    return x


with gr.Blocks() as demo:

    txt = gr.Textbox(label="What dish do you want to make today?", lines=2)
    # TODO: Call UI interaction for this
    with gr.Row():
        im = gr.Image(label="What's in your fridge?", image_mode='RGB', type='pil')

    btn = gr.Button(value="Ask Cartesan for a recipe!")
    fridge_contents = gr.Textbox(value="", label="Current Fridge Contents")
    purchase = gr.Textbox(value="", label="Ingredients Being Purchased")
    output_txt = gr.Textbox(value="", label="Recipe")
    btn.click(get_ingredients_recipe, inputs=[txt, im], outputs=[output_txt, fridge_contents, purchase])

if __name__ == "__main__":
    demo.launch()
