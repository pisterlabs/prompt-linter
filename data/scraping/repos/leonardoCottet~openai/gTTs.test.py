import openai
from gtts import gTTS
import pygame
from io import BytesIO

openai.api_key = "sua_api"
pygame.init()

def conversar(texto):
    resposta = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Suponha que você é um instrutor de Python."},
            {"role": "user", "content": texto},
        ],
        temperature=0.7,
    )
    return resposta['choices'][0]['message']['content']

def text_to_speech(text):
    tts = gTTS(text=text, lang='pt') 
    audio_file = BytesIO()
    tts.save(audio_file)
    audio_file.seek(0)
    return audio_file

while True:
    prompt = input("Usuário: ")
    resposta_chatgpt = conversar(prompt)
    print("Bot: " + resposta_chatgpt)
    
    audio_file = text_to_speech(resposta_chatgpt)
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

    # pygame sendo utilizado para aguardar a reprodução terminar antes de prosseguir
    pygame.time.wait(int(pygame.mixer.music.get_length() * 1000))
