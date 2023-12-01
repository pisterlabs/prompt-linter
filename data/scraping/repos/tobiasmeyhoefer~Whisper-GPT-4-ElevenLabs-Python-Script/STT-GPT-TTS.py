import asyncio
import websockets
import json
from openai import AsyncOpenAI
import pyaudio
import wave
import webrtcvad
import collections
import audioop
from openai import OpenAI
from dotenv import load_dotenv, dotenv_values
import base64
import shutil
import os
import subprocess

from openai import AsyncOpenAI

load_dotenv()

sclient = OpenAI()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "onwK4e9ZLuTAKqWW03F9"

vad = webrtcvad.Vad(1) 
format = pyaudio.paInt16
sample_rate = 16000
chunk_duration_ms = 30  
silence_duration_ms = 700  
channels = 1
frames = collections.deque()
threshold = 1200  
listeningIsBlocked = False



def is_installed(lib_name):
    return shutil.which(lib_name) is not None


async def text_chunker(chunks):
    """Split text into chunks, ensuring to not break sentences."""

    splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
    buffer = ""

    async for text in chunks:

        if buffer.endswith(splitters):
            yield buffer + " "
            buffer = text
        elif text.startswith(splitters):
            yield buffer + text[0] + " "
            buffer = text[1:]
        else:
            buffer += text

    if buffer:
        yield buffer + " "


async def stream(audio_stream):
    """Stream audio data using mpv player."""
    if not is_installed("mpv"):
        raise ValueError(
            "mpv not found, necessary to stream audio. "
            "Install instructions: https://mpv.io/installation/"
        )

    mpv_process = subprocess.Popen(
        ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    print("Started streaming audio")
    async for chunk in audio_stream:
        if chunk:
            mpv_process.stdin.write(chunk)
            mpv_process.stdin.flush()

    if mpv_process.stdin:
        mpv_process.stdin.close()
    mpv_process.wait()

    global listeningIsBlocked
    listeningIsBlocked = False


async def text_to_speech_input_streaming(voice_id, iterator_instance):
    """Send text to ElevenLabs API and stream the returned audio."""
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=eleven_monolingual_v1"

    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({
            "text": " ",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
            "xi_api_key": ELEVENLABS_API_KEY,
            "model_id": "eleven_turbo_v2",
        }))

        async def listen():
            """Listen to the websocket for audio data and stream it."""
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    if data.get("audio"):
                        yield base64.b64decode(data["audio"])
                    elif data.get('isFinal'):
                        break
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed")
                    break

        listen_task = asyncio.create_task(stream(listen()))

        async for text in text_chunker(iterator_instance):
            await websocket.send(json.dumps({"text": text, "try_trigger_generation": True}))

        await websocket.send(json.dumps({"text": ""}))

        await listen_task


async def chat_completion(query):
    """Retrieve text from OpenAI and pass it to the text-to-speech function."""
    theStream = await client.chat.completions.create(
      messages=[{"role": "user", "content": query}],
      stream=True,
      model="gpt-4"
    )

    async def text_iterator():
      async for part in theStream:
        delta = part.choices[0].delta
        if delta.content is not None:
          yield delta.content
        elif part.choices[0].delta.content is None:
            break
        else:
          continue

    await text_to_speech_input_streaming(VOICE_ID, text_iterator())

async def mp3ToText():
  # model = WhisperModel('tiny', compute_type="int8" )
  # segments, _ = model. transcribe ("recording.wav" )
  # text = ''.join(segment.text for segment in segments)
  audio_file= open("recording.wav", "rb")
  transcript = sclient.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
  )
  print(transcript.text)
  await chat_completion(transcript.text)

async def save_recording(audio):
    global listeningIsBlocked
    listeningIsBlocked = True
    recordedFile = "recording.wav"
    with wave.open(recordedFile, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    await mp3ToText()

async def main_loop():
    global listeningIsBlocked, speaking, audio 
    while True:
        if listeningIsBlocked:
            print("test")
            continue
        try:
          chunk = audio_stream.read(int(sample_rate * chunk_duration_ms / 1000))
          volume = audioop.rms(chunk, 2)
          is_speech = vad.is_speech(chunk, sample_rate) and volume > threshold
        except:
          audio = pyaudio.PyAudio()
          audio_stream = audio.open(format=format, channels=channels, 
                    rate=sample_rate, input=True, 
                    frames_per_buffer=int(sample_rate * chunk_duration_ms / 1000))

          num_silent_chunks_needed = int(silence_duration_ms / chunk_duration_ms)
          num_silent_chunks = 0
          speaking = False
          continue

        if speaking:
            frames.append(chunk)
            if not is_speech:
              num_silent_chunks += 1
              if num_silent_chunks > num_silent_chunks_needed:
                await save_recording(audio)
                speaking = False
                frames.clear()
            else:
              num_silent_chunks = 0
        else:
          if is_speech:
            speaking = True
            frames.append(chunk)
            num_silent_chunks = 0

asyncio.run(main_loop())


