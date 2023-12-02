import openai
import json
from datetime import datetime
with open("env.json", "r") as f:
    dictionary = json.load(f)
print(dictionary["api_key"])
api_key = dictionary["api_key"]
openai.api_key = api_key
gpt_model = "gpt-3.5-turbo"
question = "is google dialogflow better than amazon lex"
print("Request_Time: ",datetime.now().strftime("%H:%M:%S"))
print("User asking:......",question)
completion = openai.ChatCompletion.create(model=gpt_model, messages=[{"role": "user", "content": question}])
bot_response = completion.choices[0].message.content
print("Response_Time: ",datetime.now().strftime("%H:%M:%S"))
print("ChatGPT Response:...", bot_response)
