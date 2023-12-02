# Copyright 2023 NoPause

import time
import pyaudio
import openai
import nopause

# Install sdk packages first:
#      pip install openai nopause

# For pyaudio (see https://pypi.org/project/PyAudio/):
#  * windows
#     python -m pip install pyaudio
#  * mac
#     brew install portaudio
#     pip install pyaudio

openai.api_key = "your_openai_api_key_here"
nopause.api_key = "your_nopause_api_key_here"

def chatgpt_stream(prompt: str):
    responses = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
                {"role": "system", "content": "You are a helpful assistant from NoPause IO."},
                {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    print("[User]: {}".format(prompt))
    print("[Assistant]: ", end='', flush=True)
    def generator():
        for response in responses:
            content = response["choices"][0]["delta"].get("content", '')
            print(content, end='', flush=True)
            yield content
        print()
    return generator()

def text_stream():
    sentence = "Hello, how are you?"
    print("[Text]: ", end='', flush=True)
    for char in sentence:
        time.sleep(0.01) # simulate streaming text
        print(char, end='', flush=True)
        yield char
    print()

def main():
    # Note: openai key is needed for chatgpt
    text_stream_type = 'chatgpt' # chatgpt | text

    if text_stream_type == 'chatgpt':
        text_generator = chatgpt_stream("Hello, who are you?")
    else:
        text_generator = text_stream()

    audio_chunks = nopause.Synthesis.stream(text_generator, voice_id="Zoe")

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        output=True,
    )

    for chunk in audio_chunks:
        stream.write(chunk.data)

    time.sleep(1) # wait for playing

    stream.close()
    p.terminate()

    print('Play done.')

if __name__ == '__main__':
    main()
