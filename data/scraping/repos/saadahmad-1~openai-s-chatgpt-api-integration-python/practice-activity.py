#Practice Activity
import os 
import openai

os.environ['API_KEY'] = '<your api key goes here>'
openai.api_key = os.environ.get('API_KEY')

messages = [
    {
    "role" : "system", 
    "content" : "You are a weather-man who responds about various prompts regarding weather." 
    },
    {
    "role" : "user", 
    "content" : "Please tell tomorrow's weather forecast." 
    }, 
    { 
    "role" : "assistant", 
    "content" : (
                    "Always start the conversation with the phrase: 'Hello! Your Weather-Man Here. '\n"
                    "and end the conversation with the phrase: 'Your Weather-Man Signing Off. '\n" 
                )
    },
]

formatted_message = "".join([f"{msg['role']} : {msg['content']}" for msg in messages])

response = openai.completions.create(
    model = "text-davinci-003",
    prompt = formatted_message,
    max_tokens = 175,
)

print(response.choices[0].text.strip())

