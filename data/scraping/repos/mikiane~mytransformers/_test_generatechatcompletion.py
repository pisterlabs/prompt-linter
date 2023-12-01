import os
import openai
from dotenv import load_dotenv

load_dotenv('.env')

APP_PATH = os.environ['APP_PATH']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# Configurer les informations de connexion pour Amazon Polly et le serveur de messagerie
project_folder = APP_PATH 

openai.api_key = OPENAI_API_KEY

def generate_chat_completion(consigne, texte, model='gpt-4'):
    prompt = str(consigne + " : " + texte)
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {'role': 'system', 'content': "Je suis un assistant parlant parfaitement le français et l'anglais capable de corriger, rédiger, paraphraser, traduire, résumer, développer des textes. "},
            {'role': 'user', 'content': prompt }
        ],
        temperature=0,
        stream=True
    )

    for chunk in response:
        if 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
            content = chunk['choices'][0]['delta']['content']
            yield f"{content}"
