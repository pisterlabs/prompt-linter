import pyaudio
import wave
import keyboard
from openai import OpenAI
import dotenv
import datetime
import os

# Load environment variables
dotenv.load_dotenv()

client = OpenAI()

def get_question():
    # Set up parameters for recording
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024

    now = datetime.datetime.now()
    WAVE_OUTPUT_FILENAME = f"recording_{now.strftime('%Y-%m-%d_%H-%M-%S')}.wav"

    audio = pyaudio.PyAudio()

    # Start recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    print("Recording... (Press SPACE to stop)")
    frames = []

    recording = True  # Flag to control recording

    while recording:
        data = stream.read(CHUNK)
        frames.append(data)
        
        # Check if the SPACE key has been pressed
        if keyboard.is_pressed(' '):
            recording = False

    print("Finished recording")

    # Stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


    # speech to text
    audio_file = open(WAVE_OUTPUT_FILENAME, "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text"
    )

    # delete the audio file
    audio_file.close()
    os.remove(WAVE_OUTPUT_FILENAME)

    return transcript