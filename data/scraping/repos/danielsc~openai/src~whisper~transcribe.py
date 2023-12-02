# Note: you need to be using OpenAI Python v0.27.0 for the code below to work

import pyaudio
import wave
import os
import time
import asyncio

def get_temp_file():
    folder = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(folder, 'recordings', f'{int(time.time())}.wav')
    return filename

async def record():
    # record audio and return the filename

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44_100
    RECORD_SECONDS = 5

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()
    filename = get_temp_file()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename

async def transcribe(path):
    # transcribe the audio file and return the transcript
    filename = os.path.basename(path)
    print(f"Transcribing audio file {filename}")
    import openai
    audio_file= open(path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    # extract filename from path
    return filename + "| " + transcript.text

async def main():
    transcribe_future = None
    # remember start time   
    while True:
        start_time = time.time()
        filename = await record()
        print(f'Audio saved to {filename}!')
        
        if transcribe_future:
            print(f'waiting for transcription to complete ... ')
            start_time = time.time()
            text = (await transcribe_future)
            print(f'Transcript: {text}')
            transcribe_future = None
            print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

        print("starting task ...")
        start_time = time.time()
        transcribe_future = asyncio.create_task(transcribe(filename))
        print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    

if __name__ == '__main__':
    asyncio.run(main())