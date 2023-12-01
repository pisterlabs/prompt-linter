import asyncio
import websockets
import json
import openai
import base64
import shutil
import os
import subprocess

# Define API keys and voice ID
OPENAI_API_KEY = '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'
ELEVENLABS_API_KEY = 'a67ccac97e3225c39b7e6a444f68357d'
VOICE_ID = '21m00Tcm4TlvDq8ikWAM'

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY


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


async def text_to_speech_input_streaming(voice_id, text_iterator):
    """Send text to ElevenLabs API and stream the returned audio."""
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=eleven_monolingual_v1"

    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({
            "text": " ",
            "voice_settings": {"stability": 0.5, "similarity_boost": True},
            "xi_api_key": ELEVENLABS_API_KEY,
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

        async for text in text_chunker(text_iterator):
            await websocket.send(json.dumps({"text": text, "try_trigger_generation": True}))

        await websocket.send(json.dumps({"text": ""}))

        await listen_task


async def chat_completion(query):
    """Retrieve text from OpenAI and pass it to the text-to-speech function."""
    response = await openai.ChatCompletion.acreate(
        model='gpt-3.5-turbo', 
        max_tokens=150,
        messages=[

            {'role': 'user', 'content': query}
            
        ],
        temperature=1, stream=True
    )

    async def text_iterator():
        async for chunk in response:
            delta = chunk['choices'][0]["delta"]
            if 'content' in delta:
                yield delta["content"]
            else:
                break

    await text_to_speech_input_streaming(VOICE_ID, text_iterator())

def main(user_query):   
    asyncio.run(chat_completion(user_query))

   