from chat_or_use_tools import chat_or_use_tools
import time
import pyttsx3
import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.agents import load_tools, initialize_agent
import numpy as np
import pinecone
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.experimental.plan_and_execute import  load_agent_executor, load_chat_planner
import pyaudio
import wave
from pydub import AudioSegment
from hume import HumeStreamClient
from hume.models.config import  ProsodyConfig
import ssl
# import torch
# import cv2
import asyncio
import whisper
load_dotenv("keys.env")
openai.api_key = os.environ['OPENAI_API_KEY']
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 20000
RECORD_SECONDS = 10
WAVE_FILE = "output"
MP3_FILE = "test"
WHISPER_MODEL = "base"
emo = []
frames = []

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_FILE + ".wav", 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

# Open the WAV file
with wave.open(WAVE_FILE + '.wav', 'rb') as wf:
    # Get the number of frames and the sample width
    num_frames = wf.getnframes()
    sample_width = wf.getsampwidth()

    # Read the frames from the file
    frames = wf.readframes(num_frames)
    sr = wf.getframerate()
# Convert the frames to a numpy array
audio = np.frombuffer(frames, dtype=np.int16)
def save_mp3(file, data, sample_rate, batch_size = 100000):
    """Save a numpy array of audio data as an MP3 file."""
    for i in range(int(len(data)/batch_size) + 1):
        d = data[int(i * batch_size) : min(int((i + 1) * batch_size), len(data))]
        print(d)
        sound = AudioSegment(
            d.tobytes(),
            frame_rate=sample_rate,
            sample_width=data.dtype.itemsize,
            channels=1
        )
        sound.export((file + str(i) + '.mp3'), format="mp3")

# You can then use this function to save a numpy array of audio data as an MP3 file like this:
audio = np.array(audio, dtype=np.int16)

print(len(audio))
# print(audio)
save_mp3(MP3_FILE, audio, sr)

# "/content/drive/MyDrive/Hackathon/Recording.m4a"
async def go(file):
    client = HumeStreamClient(os.environ['HUME_API_KEY'])
    config = ProsodyConfig()
    
    # i = 0
    # path = os.path.join(os.getcwd(), file + str(i) + '.mp3')
        
    async with client.connect([config]) as socket:
        # while os.path.exists(path) and i < 1:
        result = await socket.send_file(filepath= 'test1.mp3')#.send_text(sample)
        emotions = result
        emo.append(emotions)    
        # i += 1
        # time.sleep(15)
        # os.remove(path)
        # path = os.path.join(os.getcwd(), file + str(i) + '.mp3')
try:
    # Create an event loop
    loop = asyncio.get_event_loop()

    # Schedule the `go` coroutine to run
    loop.run_until_complete(go(MP3_FILE))
    try:
        emo = emo[0]['prosody']['predictions'][0]['emotions']
    except:
        print(emo)
    if len(emo) == 0:
        raise FileNotFoundError(f'No data found. Probably because {MP3_FILE}0.mp3 cannot be found in the directory')
    # Close the event loop
    loop.close()

    model = whisper.load_model(WHISPER_MODEL)

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(WAVE_FILE + ".wav")
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)
    print(result.text)
    
    another_emo = []
    talk = ''
    if len(emo) > 1:
        for e in emo:
            if e['score'] > 0.5:
                another_emo.append(e['name'])
        speech = result.text + ', emotion dict: '+ str(another_emo)
        res = chat_or_use_tools(speech)
        if res != 'not speaking':
            talk = res
    else:
        speech = result.text + ", emotion dict: ['None']"
        res = chat_or_use_tools(speech)
        if res != 'not speaking':
            talk = res
except:
    talk = "I am sorry but there is a socket problem. Please try again!"
# Initialize the converter
converter = pyttsx3.init()
  
# Set properties before adding
# Things to say
# talk = 'Good morning!'
# Sets speed percent 
# Can be more than 100
converter.setProperty('rate', 200)
# # Set volume 0-1
converter.setProperty('volume', 0.5)
converter.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0')
voices = converter.getProperty('voices')

# Queue the entered text 
# There will be a pause between
# each one like a pause in 
# a sentence
print(talk)
converter.say(talk)
converter.runAndWait()