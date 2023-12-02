import discord
from discord.ext import commands as cmds
from dotenv import load_dotenv
from elevenlabslib import *
import speech_recognition as sr
import requests
import pyttsx3
import asyncio
import openai
import json
import os
import time

load_dotenv()

# settings

character = 'Rem'
keyword = 'listen'
timescoped = 5
path = 'prompts' # prompts: new, history: recall
speech_lang = 'en-US' # 'en-US', 'th-TH'
reply_lang = 'th' # 'en', 'th'

# variables

save_foldername = f'history/{character}'

if reply_lang == 'en':
    EL = str(input('use elevenlabs? (y/n): '))
    EL = True if EL == 'y' or EL == 'Y' else False
    if EL:
        eleven_name = str(input('enter eleven name: '))
        eleven_key = os.getenv('ELEVEN_KEY')
        user = ElevenLabsUser(eleven_key)
        eleven_voice = user.get_voices_by_name(eleven_name)[0]

if path == 'history':
    try:
        count = 0
        filename = os.path.join(save_foldername, f'conversation_{count}.txt')
        while os.path.exists(filename):
            count += 1
            filename = os.path.join(save_foldername, f'conversation_{count}.txt')
        if count == 0:
            raise Exception
        memIdx = int(input(f'enter memIdx for {character} (0-{count-1}): '))
    except:
        print('\nno history found (paht->prompts)\n')
        path = 'prompts'

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

openai_key = os.getenv('OPENAI_KEY')
openai.api_key = openai_key

# defualt functions

def charSet():
    if path == 'prompts':
        mount = '.txt'
        perm = 1
    else:
        mount = f'/conversation_{memIdx}.txt'
        perm = 0

    with open(f'{path}/{character}{mount}', "r", encoding='utf-8') as file:
        mode = file.read()

    if perm:
        messages  = [{"role": "system", "content": f"{mode}"}]
    else:
        messages = json.loads(mode)

    return messages

def getSuffix(save_foldername:str, messages=charSet()):
    os.makedirs(save_foldername, exist_ok=True)
    base_filename = 'conversation'
    suffix = 0
    filename = os.path.join(save_foldername, f'{base_filename}_{suffix}.txt')
    while os.path.exists(filename):
        suffix += 1
        filename = os.path.join(save_foldername, f'{base_filename}_{suffix}.txt')
    with open(filename, 'w', encoding = 'utf-8') as file:
        json.dump(messages, file, indent=4, ensure_ascii=False)
    return suffix

def trans(say):
    url = "https://translated-mymemory---translation-memory.p.rapidapi.com/get"
    querystring = {"langpair":"en|th","q":say,"mt":"1","onlyprivate":"0","de":"a@b.c"}
    headers = {
        "X-RapidAPI-Key": "1f678aafcdmshd6d45bc81882a00p18750djsnefc88f21f555",
        "X-RapidAPI-Host": "translated-mymemory---translation-memory.p.rapidapi.com"
    }
    response = requests.request("GET", url, headers=headers, params=querystring)
    data = json.loads(response.text).get('responseData').get('translatedText')
    return data       

def voiceGen(data): # wip
    url = "https://freetts.com/api/TTS/SynthesizeText"
    payload = json.dumps({
    "text": data,
    "type": 0,
    "ssml": 0,
    "isLoginUser": 0,
    "country": "Japanese (Japan)",
    "voiceType": "Standard",
    "languageCode": "ja-JP",
    "voiceName": "ja-JP-Standard-A",
    "gender": "FEMALE"
    })
    headers = {
    'authority': 'freetts.com',
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'en,th-TH;q=0.9,th;q=0.8',
    'authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjI1NDUyNDIwOTksImlhdCI6MTY4MTI0MjA4OSwiaXNzIjoia2VuIiwiZGF0YSI6eyJ1c2VybmFtZSI6IjE3Mi42OC4yNDIuMjQ4IiwiaWQiOiIxNzIuNjguMjQyLjI0OCIsImxvZ2luX3RpbWUiOjE2ODEyNDIwODl9fQ.btWVQw4ygWKTBjJ9nBhX2txZ6jlipIB49EYDNeEXZmU',
    'content-type': 'application/json',
    'cookie': '_ga=GA1.2.1262120938.1679757155; _gid=GA1.2.1415340467.1681226232; _gat=1',
    'origin': 'https://freetts.com',
    'referer': 'https://freetts.com/',
    'sec-ch-ua': '"Chromium";v="112", "Google Chrome";v="112", "Not:A-Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    res = json.loads(response.text)['data']['audiourl']
    return res
    
def textGen(messages=charSet()):
    url = "https://api.openai.com/v1/chat/completions"
    payload = json.dumps({
    "model": "gpt-3.5-turbo",
    "messages": messages,
    "max_tokens": 50
    })
    headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {openai_key}'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    data = json.loads(response.text).get('choices')[0].get('message').get('content')
    messages.append({"role": "assistant", "content": f"{data}"})
    return data

def memorize(suffix, save_foldername, messages=charSet()):
    os.makedirs(save_foldername, exist_ok=True)
    base_filename = 'conversation'
    filename = os.path.join(save_foldername, f'{base_filename}_{suffix}.txt')
    with open(filename, 'w', encoding = 'utf-8') as file:
        json.dump(messages, file, indent=4, ensure_ascii=False)

def waitListen(keyword=keyword):
    r = sr.Recognizer()
    mic = sr.Microphone()
    print("\ninitializing:")
    while True:
        with mic as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source)
            try:
                query = r.recognize_google(audio, language=speech_lang)

            except Exception:
                print('\waiting:')
                continue
            if f"{keyword}" in query.lower():
                print("true")
                return 'true'
            if ("disconnect" in query.lower()  or 
                'disconnect.' in query.lower() or 
                'ยกเลิกการทำงาน' in query.lower() or 
                'ยกเลิกการทำงาน.' in query.lower()
                ):
                return 0
            else:
                return 1
                      
def listenFor(timeout:int=30):
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source, timeout=timeout)

    return audio

# async functions

class Vchat(cmds.Cog):
    def __init__(self, bot):
        self.bot = bot

    @cmds.is_owner()
    @cmds.hybrid_command(description="voice chat with ai", usage="vchat <channel>")
    async def vchat(self, ctx, channel: discord.VoiceChannel):
        if ctx.voice_client is not None:
            return await ctx.voice_client.move_to(channel)
        
        await channel.connect()
        await ctx.send("``` voice chat on ```")
        messages = charSet()
        
        perm = 1
        while True:
            discon = 0

            if perm:
                perm = 0
                await ctx.send("> initializing")
            check = waitListen(keyword=keyword)

            if check != 'true':

                if check:
                    continue
                
                if (not check):
                    print("\nleaving:")
                    await ctx.channel.send("> leaving..")
                    await ctx.send("``` voice chat off ```")
                    await ctx.voice_client.disconnect(force=True)
                    break
            
            suffix = getSuffix(save_foldername)
            start_time = time.time()
            
            while True:
                if discon:
                    break

                await ctx.channel.send("> listening")
                print("\nlistening:")
                audio = listenFor()

                try:
                    r = sr.Recognizer()
                    query = r.recognize_google(audio, language=f"{speech_lang}")
                    messages.append({"role" : "user", "content" : query})
                    print("true")
                except:
                    if time.time() - start_time > timescoped:
                        await ctx.channel.send("> ~listening")
                        print("\n~listening:")
                        break
                    continue
                
                if "stop" in query.lower() or "stop." in query.lower() or "ยกเลิกการฟัง" in query.lower() or "ยกเลิกการฟัง." in query.lower():
                    messages.append({"role" : "assistant", "content" : "yes"})
                    memorize(suffix=suffix, save_foldername=save_foldername, messages=messages)
                    discon = 1
                    continue
                
                try:
                    await ctx.channel.send("> generating")
                    print("\ngenerating:")
                    say = textGen(messages)
                    await ctx.channel.send(say)

                    if reply_lang == 'th':
                        audio_stream = f'https://tipme.in.th/api/tts/?text={trans(say)}&format=opus'
                    elif reply_lang == 'en':
                        if EL:
                            audio_stream = eleven_voice.generate_and_stream_audio(say)
                        else:
                            audio_stream = f'https://api.streamelements.com/kappa/v2/speech?voice=Brian&text={say}'

                    source = discord.PCMVolumeTransformer(discord.FFmpegPCMAudio(audio_stream))
                    ctx.voice_client.play(source)

                    while ctx.voice_client.is_playing():
                        await asyncio.sleep(1)
                        
                    start_time = time.time()
                except Exception as e:
                    print(f"{e}")
                    print("Token limit exceededg")
                    # suffix = getSuffix(save_foldername)

            if discon:
                continue

async def setup(bot):
    await bot.add_cog(Vchat(bot))