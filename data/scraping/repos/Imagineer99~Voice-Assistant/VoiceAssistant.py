import openai
import json
import requests
import time
from elevenlabs import generate, play
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr

# Set up OpenAI API key
openai.api_key = "x"

def generate_text(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful voice assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the generated text from the response
    generated_text = response.choices[0].message.content
    return generated_text

def record_speech(duration, sample_rate=16000):
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return recording

def save_wav(audio_data, filename, sample_rate):
    sf.write(filename, audio_data, sample_rate)

def speech_to_text(audiofile):
    rec = sr.Recognizer()
    with sr.AudioFile(audiofile) as source:
        audio = rec.record(source)

    try:
        text = rec.recognize_google(audio)
    except sr.UnknownValueError:
        text = ""
    return text

def generate_audio(text):
    url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
    headers = {
        "accept": "audio/mpeg",
        "xi-api-key": "x",
        "Content-Type": "application/json"
    }
    params = {"optimize_streaming_latency": 0}

    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0
        }
    }

    response = requests.post(url, headers=headers, params=params, data=json.dumps(data))

    if response.status_code == 200:
        return response.content
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


# Run the script constantly until you close it
while True:
    # Record speech for 5 seconds
    record_duration = 5
    sample_rate = 16000
    audio_data = record_speech(record_duration, sample_rate)
    audiofile = "speech_recording.wav"
    save_wav(audio_data, audiofile, sample_rate)

    # Convert recorded speech to text
    input_text = speech_to_text(audiofile)
    print("Input Text: ", input_text)

    # Generate new text using OpenAI API
    generated_text = generate_text(input_text)

    print("Generated Text: ", generated_text)

    # Synthesize the generated text using Eleven Labs API
    audio = generate_audio(generated_text)

    # Play the synthesized audio
    play(audio)

    # Add a delay of 1 second before recording speech again
    time.sleep(1)
