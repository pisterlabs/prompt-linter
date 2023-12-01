import openai
import discord
from gtts import gTTS
#from discord.ext import commands
from discord import FFmpegPCMAudio
#from discord.utils import get
#import asyncio
#import io
import sounddevice as sd
import PySimpleGUI as sg
#import numpy as np
import scipy.io.wavfile as wav
from elevenlabs import generate, save
from elevenlabs import set_api_key
from pydub import AudioSegment
import config

##
# Danomation
# GitHub: https://github.com/danomation
# Personal Site: sussyvr.com
# Patreon https://www.patreon.com/Wintermute310
# I'm broke as hell please donate xd
##

#set your tts provider:
# valid options are "google" or "elevenlabs"
tts_provider = config.tts_provider

# instructions: Add your openai api key and bot api token
# set the target channel id for where to ask it questions with "!GPT message here"
openai.api_key = config.openai.api_key
discord_api_token = config.discord_api_token
elevenlabs_api_key = config.elevenlabs_api_key
discord_target_channel_id = config.discord_target_channel_id
set_api_key(elevenlabs_api_key)

def sendtts(message):
    if tts_provider == "elevenlabs":
        audio = generate(
        text=message,
        voice="Rachel",
        )
        save(audio, "./1.mp3")
        return "./1.mp3"
    else:
        tts = gTTS(message, tld='co.uk')
        tts.save("./1.mp3")
        return "./1.mp3"

def record_audio(filename, duration, samplerate=44100):
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=2)
    #sg.popup('Press OK when done', auto_close=True, auto_close_duration=duration,
    #         non_blocking=True, grab_anywhere=True)
    sd.wait()
    wav.write(filename, samplerate, recording)

layout = [
    [sg.Button('Record', size=(80, 80))],
]

#def connect_discord_server(transcript):
client = discord.Client(intents=discord.Intents.all())
@client.event
async def on_ready():
    window = sg.Window('Voice Proxy', layout, grab_anywhere=True, size=(290, 100))
    voice = await client.get_channel(discord_target_channel_id).connect()
    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        elif event == 'Record':
            duration = 3.0
            filename = "output.wav"
            record_audio(filename, duration)
            #sg.popup('Sent to GPT-Voice', keep_on_top=True, grab_anywhere=True)
            AudioSegment.from_wav("./output.wav").export("./output.mp3", format="mp3")
            audio_file = open("./output.mp3", "rb")
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            transcript = str(transcript.text)
            source = FFmpegPCMAudio(sendtts(transcript))
            print(transcript)
            #if voice.is_playing():
            #    await voice.stop()
            voice.play(source)  # play the source
    window.close()

client.run(discord_api_token)



