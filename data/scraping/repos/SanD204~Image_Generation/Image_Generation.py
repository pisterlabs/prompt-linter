import os
import json
import openai
import requests
import random
import time
import os

Openai_api_key = #Your_API_Key

def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        file_name = f"Image_{random.randint(0,10000)}.png"
        file_path = os.path.join(r"Output", file_name)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Image saved to {file_path}")
    else:
        print("Failed to download image")

openai.api_key ='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'

folder_path ="C:/Users/SAN/Desktop/DALL-E/Source"

for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        Response=openai.Image.create_variation(
            image=open(image_path, "rb"),
            n=5,
            size="256x256"
        )
        json_string = json.dumps(Response)
        response_dict = json.loads(json_string)
        for obj in response_dict["data"]:
            download_image(obj["url"])
            time.sleep(15)
