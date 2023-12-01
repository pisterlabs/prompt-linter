import speech_recognition as sr
import pyttsx3
import webbrowser
import pywhatkit
import subprocess


import openai
from get_env import print_env

env = print_env(['app_key'])

openai.api_key = 'Sua chave api'


model_engine = 'text-davinci-003'

audio = sr.Recognizer()
maquina = pyttsx3.init()

def listen_command():
    try:
        with sr.Microphone() as source:
            print('Escutando...')
            voz = audio.listen(source)
            comando = audio.recognize_google(voz, language='pt-br')
            comando = comando.lower()
            if 'iris' in comando:
                comando = comando.replace('iris', '')
                
    except Exception as e:
        print(f'Microfone não está ok {e}')

    return comando


def execute_command():
    comando = listen_command()
    if 'pesquise por' in comando:
        procurar = comando.replace('pesquise por', '')
        search_query = procurar 
        webbrowser.open_new_tab(f"https://www.google.com/search?q={search_query}")
        print(f"Realizando uma pesquisa no Google por '{procurar}'.")
        maquina.say(f"Realizando uma pesquisa no Google por '{procurar}'.")
        maquina.runAndWait()
    elif 'toque' in comando:
        musica = comando.replace("toque", '')
        resultado = pywhatkit.playonyt(musica)
        maquina.say(f'Tocando {musica} no youtube')
        maquina.runAndWait()

    elif 'responda' in comando or 'fale sobre' in comando or 'crie' in comando or 'o que você acha sobre' in comando:
        prompt = comando.replace('responda', '')
        prompt = comando.replace('fale sobre', '')
        prompt = comando.replace('crie', '')
        prompt = comando.replace('o que você acha sobre', '')
        completion = openai.Completion.create(
            engine = model_engine,
            prompt = prompt,
            max_tokens = 1024,
            temperature = 0.5,
        )
        reponse = completion.choices[0].text
        print(reponse)
        maquina.say(reponse)
        maquina.runAndWait()

    elif 'abra o ' in comando:
        aplicativo = comando.replace('abra o', '').strip()
        try:
            subprocess.Popen([aplicativo])
            print(f'Abrindo o aplicativo {aplicativo}...')
            maquina.say(f'Abrindo o aplicativo {aplicativo}...')
            maquina.runAndWait()
        except FileNotFoundError:
            print(f"Não foi possível encontrar o aplicativo '{aplicativo}'. "
                  f"Verifique se o nome está correto e o aplicativo está instalado.")
            maquina.say(f"Não foi possível encontrar o aplicativo '{aplicativo}'. "
                        f"Verifique se o nome está correto e o aplicativo está instalado.")
            maquina.runAndWait()

while True:
    execute_command()
    saida = input('Deseja sair?')
    if saida == 'sim':
        break
