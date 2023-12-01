from chat_or_use_tools import chat_or_use_tools
import time
import pyttsx3
import os
import openai
from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.agents import load_tools, initialize_agent
import numpy as np
import pinecone
# from langchain.chains import ConversationSummaryBufferMemory
from langchain.experimental.plan_and_execute import  load_agent_executor, load_chat_planner
import pyaudio
import wave
from pydub import AudioSegment
from hume import HumeStreamClient
from hume.models.config import  ProsodyConfig
import websockets
# import torch
# import cv2
import asyncio
import whisper
load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 20000
RECORD_SECONDS = 10
WAVE_FILE = "output"
MP3_FILE = "test.mp3"
WHISPER_MODEL = "base"


def recording():
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

    

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("* done recording")

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
    return np.array(np.frombuffer(frames, dtype=np.int16), dtype=np.int16), sr

# audio, sr = recording()
def save_mp3(file, data, sample_rate, batch_size = 30000):
    """Save a numpy array of audio data as an MP3 file."""
    d = data[int((len(data) - batch_size) / 2) : int((len(data) + batch_size)/2)]
    sound = AudioSegment(
        d.tobytes(),
        frame_rate=sample_rate,
        sample_width=data.dtype.itemsize,
        channels=1
    )
    sound.export(file, format="mp3")

# save_mp3(MP3_FILE, audio, sr)
# emo = []
async def go(file, emo):
    
    client = HumeStreamClient(os.environ['HUME_API_KEY'])
    config = ProsodyConfig()
    
        
    async with client.connect([config]) as socket:
        try:
            result = await socket.send_file(filepath= file)#.send_text(sample)
            emotions = result
            emo.append(emotions)    
        except websockets.exceptions.ConnectionClosedOK:
            pass
# try:
#     # Create an event loop
#     loop = asyncio.get_event_loop()

#     # Schedule the `go` coroutine to run
#     loop.run_until_complete(go(MP3_FILE))
#     try:
#         emo = emo[0]['prosody']['predictions'][0]['emotions']
#     except:
#         print(emo)
#     if len(emo) == 0:
#         raise FileNotFoundError(f'No data found. Probably because {MP3_FILE} cannot be found in the directory')
#     # Close the event loop
#     loop.close()


def speech_to_text(audio, emo, memory = None):
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
        res = chat_or_use_tools(speech, memory)
        if res != 'not speaking':
            talk = res
    else:
        speech = result.text + ", emotion dict: ['None']"
        res = chat_or_use_tools(speech, memory)
        if res != 'not speaking':
            talk = res
    return talk
    # talk = speech_to_text(audio)
# except:
#     talk = "I am sorry but there is a socket problem. Please try again!"

def text_to_speech(talk):
    # Initialize the converter
    converter = pyttsx3.init()
    
    # Set properties before adding
    converter.setProperty('rate', 200)
    # # Set volume 0-1
    converter.setProperty('volume', 0.5)
    converter.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0')

    # Queue the entered text 
    # There will be a pause between
    # each one like a pause in 
    # a sentence
    converter.say(talk)
    converter.runAndWait()

# text_to_speech(talk)

def checking():
    print(os.environ['OPENAI_API_KEY'])
    audio, sr = recording()
    save_mp3(MP3_FILE, audio, sr)
    emo = []    
    # try:
    # Create an event loop
    loop = asyncio.get_event_loop()

    # Schedule the `go` coroutine to run
    loop.run_until_complete(go(MP3_FILE, emo))
    print(emo)
    emo = emo[0]['prosody']['predictions'][0]['emotions']
    if len(emo) == 0:
        raise FileNotFoundError(f'No data found. Probably because {MP3_FILE} cannot be found in the directory')
    # Close the event loop
    loop.close()    
    talk = speech_to_text(audio)
    # except:
    #     time.sleep(60)
    #     checking()
    text_to_speech(talk)
    # checking()

def asking():
    llm = OpenAI(temperature=0)
    # memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=2000)
    # memory.save_context({"input": "Hello"}, {"output": "What's up"})
    # print(memory.load_memory_variables({}))
    audio, sr = recording()
    emo = []    
    try:
        talk = speech_to_text(audio, emo)
        text_to_speech(talk)
    except:
        print("Something is wrong try again later!")