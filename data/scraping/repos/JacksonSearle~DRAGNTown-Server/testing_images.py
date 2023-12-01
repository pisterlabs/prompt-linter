import openai
from dotenv import load_dotenv
import requests
import datetime
import os
import json
import time


from config import path
openai.api_key_path = (path + ".env")
load_dotenv(path)

# Maximum number of API calls per day
MAX_API_CALLS = 50


def download_image(url, save_path):
    save_path = 'images/' + save_path + '.png'
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Image downloaded and saved to {save_path}")
    else:
        print(f"Failed to download the image. Status code: {response.status_code}")

def generate_image(prompt, save_path):
    # Load or initialize the count and date
    filename = path + 'prompts/api_calls.json'
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            call_count = data['image_count']
            last_call_date = datetime.datetime.strptime(data['date'], '%Y-%m-%d').date()
    else:
        call_count = 0
        last_call_date = datetime.date.today()

    # Check if it's a new day
    if datetime.date.today() != last_call_date:
        # If it's a new day, reset the count
        call_count = 0
        last_call_date = datetime.date.today()

    if call_count < MAX_API_CALLS:

        try:
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
        except openai.error.ServiceUnavailableError as e:
            print('Server is busy, trying again in 3 seconds.')
            time.sleep(3)
            return generate_image(prompt, save_path)

        url = response['data'][0]['url']

        # Increment and save the count
        call_count += 1
        with open(filename, 'w') as f:
            json.dump({'text_count': data['text_count'], 'image_count': call_count, 'date': str(last_call_date)}, f)
        download_image(url, save_path)
    else:
        print(f"API call limit of {MAX_API_CALLS} per day reached. Please wait until tomorrow.")

# Usage
# generate_image('A lego robot', 'lego_robot')