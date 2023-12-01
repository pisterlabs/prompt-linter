import pyaudio
import wave
import subprocess
import openai
import json
import base64

with open("config.json") as file:
    config = json.load(file)

openai.api_key = config["api_key"]

audio_filename = input("Wav filename: ")

prompt = "Transcribe the following audio:\n"

def convert_to_base64(audio_data):
    base64_audio = base64.b64encode(audio_data).decode('utf-8')
    return base64_audio

def get_wav_duration(filename):
    with wave.open(filename, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        
    return duration
    
def read_audio_file(filename):
    with wave.open(filename, 'rb') as wav_file:
        audio_data = wav_file.readframes(-1)
        
    return audio_data

# Record audio using pyaudio
def record_audio(filename, duration):
    chunk = int(config["chunk"])
    rate = int(config["rate"])
    channels = int(config["channels"])
    
    format = pyaudio.paInt16
    capture_duration = duration
    
    p = pyaudio.PyAudio()
    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)
                    
    print("[+] Started recording")
    
    frames = []
    for i in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
        print(f"{i}/{int(rate / chunk * duration)}")
        
    print("[+] Finished recording")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Transcribe audio using OpenAI
def transcribe_audio(audio_filename):
    # Define your transcription prompt
    transcription_prompt = "Transcribe the following audio:"

    # Read the audio file and convert it to base64
    audio_data = read_audio_file(audio_filename)
    base64_audio = convert_to_base64(audio_data)

    # Generate the transcription using OpenAI API
    response = openai.Completion.create(
        engine=config["engine"],
        prompt=transcription_prompt + '\n\nAudio: ' + base64_audio,
        max_tokens=int(config["max_tokens"]),
        temperature=0.6,
        n = 1,
        stop=None
    )

    # Extract and return the transcribed text
    transcription = response.choices[0].text.strip()
    return transcription

# Calculate the duration of the audio file
duration = get_wav_duration(audio_filename)

# Record audio and transcribe it
record_audio(audio_filename, duration)
transcription = transcribe_audio(audio_filename)

print(f"Transcription:\n{transcription}")
