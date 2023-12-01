import streamlit as st
import os, io
import json, time
import openai
import base64, requests
from st_files_connection import FilesConnection
from datetime import datetime


api_key = st.secrets["openai_key"]
conn = st.connection('gcs', type=FilesConnection)

def get_imgname():
    # Generate a unique filename using the current datetime
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unique_filename = f"{current_time}_image.jpeg"
    return unique_filename

def save_img(image, path):
    base64_image = base64.b64encode(image.getvalue()).decode("utf-8")
    with conn.open("gs://" + path, 'wb') as f:
        f.write(base64.b64decode(base64_image))

def add_metadata(path, response, imgname):
    with conn.open(path, 'r', encoding='utf-8') as f:
        data = f.read()
    data = data + "\n" + "PIC: "+ imgname + '\n' + response
    with conn.open(path, 'w', encoding='utf-8') as f:
        f.write(data)

def answer_pic(image, prompt):
# Path to your image
        # Getting the base64 string
    base64_image = base64.b64encode(image.getvalue()).decode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            ]
        }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response.json()



image = st.camera_input("Take a picture")
pro = "What's in this image? Answer in json only. If it's meal, list it's components in structure main_dish, side_dish, garnish, topping, other. There can be multiple subitems for each component, if it fits the category. List up to 5 alternative components sorted by probability of correct identification. If no item of component type is present, leave it blank with empty dictionary. Estimate nutritional values (calories, fat, protein, carbs) for each component. Provide one number as your best guess. Keep it simple and short. No explanations."

prompt = st.text_area("What's your question?", value = pro)

if st.button("Ask"):
    if image:
        st.image(image)
        response = answer_pic(image, prompt)['choices'][0]['message']['content']
        st.write(response)
        imgname = get_imgname()
        path = 'food-bro/img-captures/' + imgname
        save_img(image, path)
        add_metadata('food-bro/img_data.json', response, imgname)
        st.write("Image saved: ", path)
    else:
        st.write("Please take a picture")

