import os
import io
import pyaudio
import wave
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
import openai
import subprocess

# Set your API keys
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "구글.json"
with open('openai_session.txt', 'r') as file:
    api_key = file.read().strip()

openai.api_key = api_key


# Function to record audio using PyAudio
def record_audio(file_path, record_seconds=5):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("Recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


# Function to transcribe audio using Google Speech-to-Text API
def transcribe_audio(file_path, language_code="ko-KR"):
    client = speech.SpeechClient()

    with io.open(file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
    )

    response = client.recognize(config=config, audio=audio)

    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript

    return transcript


# Function to interact with ChatGPT using OpenAI API
def chat_with_gpt(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()


# Function to convert text to speech using Google Text-to-Speech API
def text_to_speech(text, output_file="output.wav"):
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
        name="ko-KR-Wavenet-A",
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open(output_file, "wb") as out:
        out.write(response.audio_content)


# Function to play audio file
def play_audio(file_path):
    os.system(f"aplay {file_path}")


# Function to start voice.py
def start_voice_chat():
    try:
        subprocess.run(["python3", "/home/pi1/weeb1/start_voice.py"])
    except Exception as e:
        print(f"Error starting voice chat: {e}")


# Main function
def main():
    while True:
        # Record audio
        audio_file_path = "input.wav"
        record_audio(audio_file_path)

        # Transcribe audio to text
        user_input = transcribe_audio(audio_file_path)

        if not user_input:
            start_voice_chat()
            continue

        print("You:", user_input)

        # Chat with GPT
        gpt_response = chat_with_gpt(f"You: {user_input}\nBot:")
        print("Bot:", gpt_response)

        # Convert GPT response to speech
        text_to_speech(gpt_response)

        # Play the response
        play_audio("output.wav")


if __name__ == "__main__":
    main()
