import openai
import anthropic
import streamlit as st
import requests
from PIL import Image
import io
from scripts.key import openai_key, anthropic_key, hf_key

def get_model_list():
    return tuple(models.keys())

def get_model(model_name):
    return models[model_name]

models = {
    "GPT-3.5": "gpt-3.5-turbo",
    "GPT-4 (use your own key)": "gpt-4",
    "Claude Instant v1.3": "claude-instant-v1",
    "Claude v1.3": "claude-v1",
    "SD v2": "stabilityai/stable-diffusion-2",
    "SD v2.1": "stabilityai/stable-diffusion-2-1",
    "SD v1.5": "runwayml/stable-diffusion-v1-5",
    "Openjourney (SD)": "prompthero/openjourney",
}


# generate a response
def gpt_generate_response(model, temperature):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=st.session_state['messages'],
        temperature=temperature
    )
    response = completion.choices[0].message.content
    # print(st.session_state['messages'])
    # total_tokens = completion.usage.total_tokens
    # prompt_tokens = completion.usage.prompt_tokens
    # completion_tokens = completion.usage.completion_tokens
    return response #, total_tokens, prompt_tokens, completion_tokens

def gpt_generate_response_stream(model, temperature):
    response = openai.ChatCompletion.create(
        model=model,
        messages=st.session_state['messages'],
        temperature=temperature,
        stream=True
    )
    return response

def claude_generate_response_stream(model, temperature, max_tokens_to_sample=200):
    c = anthropic.Client(anthropic_key)
    prompt = gpt_to_claude()
    response = c.completion_stream(
            prompt=prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            max_tokens_to_sample=max_tokens_to_sample,
            temperature = temperature,
            model=model,
            stream=True,
        )
    return response

def claude_generate_response(model, temperature, max_tokens_to_sample=200):
    c = anthropic.Client(anthropic_key)
    prompt = gpt_to_claude()
    response = c.completion(
            prompt=prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            max_tokens_to_sample=max_tokens_to_sample,
            temperature = temperature,
            model=model,
        )
    return response["completion"]

def gpt_to_claude():
    ret = []
    last_role = ''
    for m in st.session_state['messages']:
        if m['role'] == last_role:
            ret[-1] += (' ' + m['content'])
        else:
            if m['role'] == 'system' or m['role'] == 'user':
                ret.append(anthropic.HUMAN_PROMPT + ' ' + m['content'])
            else:
                ret.append(anthropic.AI_PROMPT + ' ' + m['content'])
    ret.append(anthropic.AI_PROMPT)
    ret = ''.join(ret)
    return ret

def query(payload, API_URL):
    headers = {"Authorization": hf_key}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return response.status_code
    return response.content

def sd_generate_response(model):
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    image_bytes = query({
        "inputs": st.session_state['messages'][-1]['content'],
        "wait_for_model": True,
    },
    API_URL)
    if isinstance(image_bytes, int):
        return image_bytes
    image = Image.open(io.BytesIO(image_bytes))
    return image