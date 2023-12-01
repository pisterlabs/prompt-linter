import os 
import openai

os.environ['API_KEY'] = '<your api key goes here>'
openai.api_key = os.environ.get('API_KEY')

messages = [
    {"role" : "system", "content" : "You are a friendly assistant, who reponses in English"},
    {"role" : "user" , "content" : "How many branches does the bank have in Massachusetts?"},
    {"role" : "assistant" , "content" : "You always start a conversation with the phrase: 'Hiya! '"},
]

formatted_messages = "".join([f"{msg['role']}: {msg['content']}" for msg in messages])

response = openai.completions.create(
    model = "text-davinci-003",
    prompt = formatted_messages,
    max_tokens = 75
)

print(response.choices[0].text.strip())