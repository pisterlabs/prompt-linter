from elevenlabs import clone, generate, play, set_api_key, voices, save
from IPython.display import Audio
import openai
from subprocess import run
import os, datetime, time

openai.api_key  = ""  # insert the OpenAI API key here
set_api_key('') # Insert the ElevenLabs


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def generate_audio(prompt):

    parent = './audios_response'
    os.makedirs(parent, exist_ok=True)

    prompt = prompt + ' Use at most 20 words'
    chatgpt_resp = get_completion(prompt)
    # chatgpt_resp = chatgpt_resp + ' Use at most 20 words'
    # print(chatgpt_resp)

    audio_response = generate(text=chatgpt_resp, model="eleven_monolingual_v1", voice='Fritz')
    # saving audio file in wav format.
    now = datetime.datetime.now()
    time_ = now.time()
    time_var = time_.isoformat().split('.')[0]
    time_var = time_var.replace(':', '_')
    audio_name = parent + '/chatgpt_response_' + time_var + '.wav'
    save(audio_response, audio_name)

    return audio_name