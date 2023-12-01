import requests
import json
from datetime import datetime
import openai
import env


def call_tele(msg):
    requests.get('https://api.telegram.org/bot' + env.tl_key + '/sendMessage?chat_id=' + env.chat_id + '&parse_mode=Markdown&text=' + msg)

openai.api_key = env.openai_key

now = requests.get(f"https://apiadvisor.climatempo.com.br/api/v1/weather/locale/{env.city_id}/current?token={env.clima}").json()

all_temps = requests.get(f"http://apiadvisor.climatempo.com.br/api/v2/forecast/temperature/locale/{env.city_id}/hours/168?token={env.clima}").json()['temperatures']
# Select a few hours and parse data
temps = [[i['date'].split(" ")[1][:5], i['value']] for i in all_temps]
temps_table = [temps[i] for i in range(0,len(temps[:20]),4)]

prompt = f"""
WRITE YOUR PROMPT HERE
INSERT DATA {now} AND {temps_table}
TIP: USE CHATGPT TO GENERATE IT
"""

gpt_msg = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=1,
    max_tokens=512,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)
message = gpt_msg['choices'][0]['message']['content']

#print(message)  # Debug 
call_tele(message)  # Send message
