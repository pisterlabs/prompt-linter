import os
import requests
import json
import pinecone
from dotenv import load_dotenv
from openai import OpenAI
import openai

load_dotenv()
openai_api_key : str = os.getenv("OPENAI_API_KEY")


url = "https://api.openai.com/v1/embeddings"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai_api_key}"  
}


def get_openai_completion(input_text: str) -> str:
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": input_text}],
        "temperature": 0.7
    }

    url = "https://api.openai.com/v1/chat/completions"

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        result = response.json()
        completion = result['choices'][0]['message']['content']
        return completion
    else:
        print("Request failed with status code:", response.status_code)
        print("Error message:", response.text)
        return None 



def generate_embedding(data):
    print(data)
    data = "name: " + data["name"] + " price: " + str(data["price"]) + " brand: " + data["brand"] +  "category: " + ' '.join(data["category"])
    payload = {
        "input": data,
        "model": "text-embedding-ada-002"
    }
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        embeddings_data = response.json()
    else:
        print("Failed to fetch embeddings. Status code:", response.status_code)
        return None
    return embeddings_data

def generate_embedding_for_prompt(data):
    payload = {
        "input": data,
        "model": "text-embedding-ada-002"
    }
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        embeddings_data = response.json()
    else:
        print("Failed to fetch embeddings. Status code:", response.status_code)
        return None
    return embeddings_data