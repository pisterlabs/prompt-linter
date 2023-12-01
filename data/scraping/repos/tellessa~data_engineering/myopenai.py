import os
import openai
import requests

print(openai.organization)
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()

# Set the API endpoint and your API key
endpoint = "https://api.openai.com/v1/images/generations"
api_key = os.getenv("OPENAI_API_KEY")

# Set the model and prompt
model = "image-alpha-001"
prompt = "Generate an image of a kitten wearing a hat"

# Set the data for the request
data = {
    "model": model,
    "prompt": prompt,
    "num_images": 1,
    "size": "1024x1024",
    "response_format": "url"
}

# Set the headers for the request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Send the request
response = requests.post(endpoint, json=data, headers=headers)

# Print the response
print(response.json())
