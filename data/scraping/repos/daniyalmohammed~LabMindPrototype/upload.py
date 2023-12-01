from openai import OpenAI
import firebase_admin
from firebase_admin import credentials, db
import json
import shutil
import os
import subprocess
import base64
import requests
import uuid

# Initialize OpenAI and Firebase Admin
client = OpenAI(api_key='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')
cred = credentials.Certificate('./credentials.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://labmindprototype-default-rtdb.firebaseio.com/'
})


# OpenAI API Key
api_key = "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"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "./Images/8-S1-no_area-100k-ordered.jpg"
path_parts = image_path.split('/')

# Get the last element, which is the file name
file_name = path_parts[-1]

# Getting the base64 string
base64_image = encode_image(image_path)

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
          "text": '''Give me JSON and ONLY THE JSON no other punctuation except the json that matches this format based on the image: 
           {
                "microscope_model": "SU8000",
                "EHT_kV": 20.0,
                "working_distance_mm": 9.6,
                "magnification": 50000,
                "signal_type": "SE(U)",
                "sample_description": "Pd nanoparticles on carbon, Dataset 1 (ordered)",
                "DOI": "10.6084/m9.figshare.11783661",
                "project_association": "AnanikovLab.ru",
                "image_format": "tif",
                "image_type": "SEM_image",
                "field_of_study": "nanomaterials",
                "file_name": "532-S1-A50-50k-ordered.tif"
                }

          '''
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


if response.status_code == 200:
    data = response.json()  # Get the JSON data from the response
    
    # Access the message content from the response data
    resp = data['choices'][0]['message']['content']
    print(resp)
    
    # Upload the response data to the Firebase Realtime Database
    root_ref = db.reference('/')
    child_ref = root_ref.child(str(uuid.uuid4()))
    resp1 = json.loads(resp)
    resp1["file_name"] = file_name
    
    # Use the set() method on the child reference to add data under the specified key
    child_ref.set(resp1)
    
    print('Data uploaded successfully.')
else:
    print(f"API request failed with status code {response.status_code}")
