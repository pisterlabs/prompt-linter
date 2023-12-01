import os
import re
import openai
import whisper
import torch
import numpy as np
import pyaudio
import socket
import wave

openai.api_key = None
import guidance

# connect to a chat model like GPT-4 or Vicuna
gpt4 = guidance.llms.OpenAI("gpt-3.5-turbo")
# vicuna = guidance.llms.transformers.Vicuna("your_path/vicuna_13B", device_map="auto")
valid_weapons = ["sword", "axe", "mace", "spear", "bow", "crossbow"]
experts = guidance('''
{{#system~}}
Generate a character profile for an RPG game in JSON format.
{{~/system}}


{{#user~}}
{{query}}
{{~/user}}

{{#assistant~}}
{{gen 'answer' temperature=0 max_tokens=500}}
{{~/assistant}}
''', llm=gpt4)

print(experts(query='Great, now generate a character profile for any input.'))
# model = whisper.load_model("base", "cuda")


# def transcribe(audio):
#     audio = whisper.load_audio(audio)
#     audio = whisper.pad_or_trim(audio)
#     mel = whisper.log_mel_spectrogram(audio).to(model.device)
#     _, probs = model.detect_language(mel)
#     global language
#     language = max(probs, key=probs.get)
#     options = whisper.DecodingOptions(fp16=False)
#     result = whisper.decode(model, mel, options)
#     return result.text


# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 44100
# CHUNK = 4096

# ctx = [
#     {
#         "role": "system",
#         "content": """You are Jarvis, a artificial intelligence with humor and great reasoning capacity. You are not an AI language model.

# for any real-time information or fact, write a python script to google search using beasutifulsoup it and return the first result
# For any tasks, write a python script to do them without asking for confirmation. The cursrent date is 2023-4-4.""",
#     }
# ]

# audio = pyaudio.PyAudio()
# # start Recording
# stream = audio.open(
#     format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
# )
# timer = 0
# frs = []
# print("recording...")
# try:
#     while True:
#         stream.start_stream()
#         data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
#         frs.extend(data)
#         timer += 1
#         if timer % 50 == 0:
#             timer = 0
#             stream.stop_stream()
#             # stream.close()
#             wavefile = wave.open("output.wav", "wb")
#             wavefile.setnchannels(CHANNELS)
#             wavefile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
#             wavefile.setframerate(RATE)
#             wavefile.writeframes(b"".join(frs))
#             wavefile.close()
#             frs = []
#             print("transcribing...")
#             txt = transcribe(
#                 r"C:\Users\jetjo\OneDrive\Documents\scripts\chatgpt\output.wav"
#             )
#             print(txt)
#             print("chatting...")
#             ctx.append({"role": "user", "content": txt})
#             print(ctx)
#             if txt not in ["Thank you for watching.", "Thank you for watching!"]:
#                 response = openai.ChatCompletion.create(
#                     model="gpt-3.5-turbo", messages=ctx, temperature=0.9,
#                 )
#                 try:
#                     exec(
#                         re.sub(
#                             "python",
#                             "",
#                             response.choices[0].message["content"].split("`")[3],
#                             count=1,
#                         )
#                     )
#                 except:
#                     print(response.choices[0].message["content"])
#                 ctx.append(
#                     {
#                         "role": "system",
#                         "content": response.choices[0].message["content"],
#                     }
#                 )

# except KeyboardInterrupt:
#     pass

# print("Shutting down")
# # s.close()
# stream.close()
# audio.terminate()

# # text = """Here's a Python script that googles the current temperature in New Brunswick, New Jersey and returns the first result:

# # ```python
# # import requests
# # from bs4 import BeautifulSoup

# # query = 'temperature in New Brunswick, New Jersey'
# # url = f'https://www.google.com/search?q={query}'

# # headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
# # res = requests.get(url, headers=headers)

# # soup = BeautifulSoup(res.text, 'html.parser')
# # temp = soup.select('#wob_tm')[0].text.strip()

# # print(f"The current temperature in New Brunswick, New Jersey is {temp}°F.")
# # ```"""
# # exec(re.sub("python","",text.split("`")[3],count=1))
# # print(re.sub("python","",text.split("`")[3],count=1))
