import os 
import openai

os.environ['API_KEY'] = '<your api key goes here>'
openai.api_key = os.environ.get('API_KEY')

response = openai.completions.create(
    model = "text-davinci-002",
    prompt = "Once upon a time, when the world was young,",
    max_tokens = 75
)

print(response.choices)
