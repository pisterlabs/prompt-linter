import openai
import requests

# Configurer les clés d'API Mattermost et OpenAI
MATTERMOST_URL = "https://mattermost.example.com"
MATTERMOST_TOKEN = "your_mattermost_token"
OPENAI_API_KEY = "your_openai_api_key"

# Configurer les paramètres de la requête GPT-3
openai.api_key = OPENAI_API_KEY
prompt = (f"Interagir avec ChatGPT via Mattermost")
completions = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=1024, n=1,stop=None,temperature=0.5)

# Extraire la réponse de GPT-3
message = completions.choices[0].text

# Préparer la requête pour Mattermost
headers = {
    "Authorization": f"Bearer {MATTERMOST_TOKEN}",
    "Content-Type": "application/json",
}
data = {
    "channel_id": "your_channel_id",
    "message": message,
}

# Envoyer la réponse à Mattermost
requests.post(f"{MATTERMOST_URL}/api/v4/posts", headers=headers, json=data)
