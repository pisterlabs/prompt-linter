import os
import openai
import sounddevice as sd
from scipy.io.wavfile import write



def transcribe_audio(audio_file_name="input_audio_whisper.wav"):
    audio = open(audio_file_name, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio)
    return transcript


def record_audio(fs=44100, seconds=5, audio_file_name="input_audio_whisper.wav"):
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write(audio_file_name, fs, myrecording)  # Save as WAV file


audio_name = "input_audio_whisper.wav"
record_audio(audio_file_name=audio_name)
text = transcribe_audio(audio_name)["text"]
print(text)
