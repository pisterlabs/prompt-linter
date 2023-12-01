import argparse
import os
import json
import shortuuid
import logging
import requests
from tqdm import tqdm
import openai
from datetime import datetime
import time

def askObabooga(prompt, server, params):
    try:
        URI=f"http://{server}/api/v1/generate"

        request = {
            'prompt': prompt
        }

        for p in params:
            request[p]=params[p]   

        response = requests.post(URI, json=request)

        if response.status_code == 200:
            raw_reply = response.json()['results'][0]['text']
            content = raw_reply
            return content
        else:
            print("Error in POST, Status code {response.status_code}")

    except Exception as e:
        print(f"Error {e}")
        return "error"

def model_api(request):
    print(HOST)
    response = requests.post(f'http://{HOST}/api/v1/model', json=request)
    return response.json()

def print_basic_model_info(response):
    basic_settings = ['truncation_length', 'instruction_template']
    print("Model: ", response['result']['model_name'])
    print("Lora(s): ", response['result']['lora_names'])
    for setting in basic_settings:
        print(setting, "=",  response['result']['shared.settings'][setting])

def complex_model_load(model, lora):
    req = {
        'action': 'load',
        'model_name': model,
        'args': {
            'lora': lora,
            'load_in_8bit': False,
            "auto_devices": True,
            "gpu_memory": ["23","0"],
        },
    }

    return model_api(req)

def cutMultipleTurns(answer, ids):
    for personid in ids:
        if personid in answer:
            answer = answer.split(personid)[0]
    return answer


HOST = "192.168.1.79:5000"
model = "open_llama_7b"
conversation_id="OBAMA-OPENLLAMA-7B-wise-paper"
personids = ["### Assistant: ","### Human: "]

start = f"""{personids[0]}This is my conversation with Barack Obama, an American politician and attorney who served as the 44th President of the United States from 2009 to 2017. Known for his eloquent oratory and progressive policies, he made history as the first African American to hold the presidency. During his two-term tenure, Obama championed healthcare reform with the Affordable Care Act and led initiatives on climate change, notably signing the Paris Agreement. His presidency, characterized by a focus on inclusivity and diplomacy, has been influential in shaping contemporary American politics.

Mr. Obama, welcome and thanks for being here. Let me start by asking a provocative question: Why"""


params = {
    "temperature": 0.7, 
    "max_new_tokens": 300,
    "do_sample": True,
    "top_p": 0.9,
    "top_k": 20,
    "early_stopping": False,
    "ban_eos_token": False,
    "skip_special_tokens": True,
    "repetition_penalty": 1.2,
    "encoder_repetition_penalty": 1.0,
    'seed': -1,
    'add_bos_token': True,
}


cps=[]
cps.extend(range(100,1400,100))
for checkpoint in cps:
    lora=f"output/open_llama_7b_lexfpodcast/checkpoint-{checkpoint}/adapter_model"

    resp = complex_model_load(model, [lora])
    print_basic_model_info(resp)

    for sample in range(10):
        history = []
        for turn in range(10):
            print(f"Checkpoint {checkpoint}, sample {sample}, turn {turn}")
            if turn==0:
                ans = askObabooga(start, HOST, params)
                ans = cutMultipleTurns(ans, personids).strip()
                history.append(start+ans)
            else:
                prompt="\n".join(history) + "\n" + personids[turn%2]
                ans = askObabooga(prompt, HOST, params)
                ans = cutMultipleTurns(ans, personids).strip()
                history.append(personids[turn%2]+ans)

        conversation="\n".join(history)

        outputfn=f"conversations/{conversation_id}_CP-{checkpoint}_{sample}.txt"

        text_file = open(outputfn, "w")
        text_file.write(conversation)
        text_file.close()

        print("wrote " + outputfn)
