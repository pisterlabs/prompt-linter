from tkinter import *
import openai
import requests

with open('key.txt', 'r') as f:
    openai.api_key = f.read()[:-1]

def AI_input(text, from_begin):
    if from_begin:
        request = text.get(1.0, 'end-1c')
    else:
        request = text.get(1.0, text.index(INSERT)).splitlines()[-1]
    return request

def GPTJ(text, from_begin):
    request = AI_input(text, from_begin)
    payload = {
        "model" : "GPT-J-6B",
        "context": request,
        "token_max_length": 512,
        "temperature": 0.8,
        "top_p": 0.9,
        "stop_sequence" : None
    }
    result = requests.post("http://api.vicgalle.net:5000/generate", params=payload).json()['text']
    text.insert(text.index(INSERT), result)
    text.update()


def Codex(text, stop, from_begin):
    request = AI_input(text, from_begin)
    result = Codex_answer(request + '\n', stop)
    text.insert(text.index(INSERT), '\n' + result)
    text.update()

def Codex_answer(string, stop=None):
  response = openai.Completion.create(
    engine="davinci-codex",
    prompt=string,
    temperature=0.2,
    max_tokens=512,
    top_p=1,
    frequency_penalty=0.5,
    presence_penalty=0,
    stop=stop
  )
  return response['choices'][0]['text']
