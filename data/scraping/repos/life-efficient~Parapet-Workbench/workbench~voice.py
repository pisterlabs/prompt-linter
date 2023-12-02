# %%
import sounddevice as sd
from pydub import AudioSegment
from pydub.utils import make_chunks
from elevenlabs import generate, play, save, stream, voices
from time import time
import numpy as np
import os
from openai import OpenAI
from workbench.utils.set_keys import set_or_get_api_key

set_or_get_api_key(api_name="elevenlabs")

client = OpenAI()

def speak(text, voice="Nicole", _stream=True, play=True, save_fp=None):
    audio = generate(
        text=text,
        stream=True if not save_fp else False,
        # stream=_stream,
        voice=voice,
    )
    if play:
        stream(audio)
    if save_fp:
        save(audio, save_fp)

    # play(audio)
    # save audio file

    # save audio as mp3
    return audio


def voice_to_text(duration=15):

    # Define the recording parameters
    fs = 44100  # Sample rate (samples per second)

    print("Listening...")

    # Start recording audio until the user hits Enter
    audio_data = []
    with sd.InputStream(samplerate=fs, channels=1, dtype=np.int16) as stream:
        start_time = time()
        while time() - start_time < duration:
            # Read a chunk of audio data
            chunk, overflowed = stream.read(fs)
            audio_data.append(chunk)

            # Check if the user hit Enter
            # if input("") == "":
            #     break

    print("Thinking...")

    # Convert the recorded audio data into a NumPy array
    audio_data = np.concatenate(audio_data)

    audio = AudioSegment(audio_data.tobytes(), frame_rate=fs,
                         sample_width=2, channels=1)

    temp_audio_filename = "temp.mp3"
    audio.export(temp_audio_filename, format="mp3")

    with open(temp_audio_filename, 'rb') as f:
        text = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        ).text

    os.remove(temp_audio_filename)

    return text


if __name__ == "__main__":
    # audio = generate(
    #     text="Hi! My name is Bella, nice to meet you!",
    #     voice="Bella",
    #     model="eleven_monolingual_v1"
    # )

    text = "Hello there, I'm your new AI assistant"
    speak(text, save=False)
    # audio = generate(text, voice="Nicole")
    # stream(audio)

# %%


# audio = generate(
#     text="Hello there, I'm your new AI assistant",
#     voice="Bella",
#     model='eleven_multilingual_v1'
# )

# play(audio)
