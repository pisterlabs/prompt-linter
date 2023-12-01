import requests
from dotenv import load_dotenv
import os
import cohere
import psycopg2
import requests

load_dotenv()
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
COCKROACH_USERNAME = os.getenv('htn')
COCKROACH_PASSWORD = os.getenv('X4Vc6r3tQ8jngyPzLGgIdA')

co = cohere.Client(COHERE_API_KEY)

conn = psycopg2.connect(
    user= COCKROACH_USERNAME,
    password=COCKROACH_PASSWORD,
    database='db',
)

cur = conn.cursor()

document_name = 'Your_Document_Name'

cur.execute("SELECT id FROM documents WHERE name = %s", (document_name,))
document_id = cur.fetchone()[0]

cur.close()
conn.close()

url = "https://api.cohere.ai/v1/chat"
payload = {
    "message": "Can you give me a global market overview of solar panels?",
    "temperature": 0.3,
    "stream": False,
    "chat_history": [
        {
            "user_name": "Chatbot",
            "message": "How can I help you today?"
        },
    ],
    "prompt_truncation": "OFF",
    "search_queries_only": False,
    "documents": [{"id": str(document_id)}]
}

headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": "Bearer " + COHERE_API_KEY,
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)
