import speech_recognition as sr
r = sr.Recognizer()
with sr.Microphone() as source:
    r.pause_threshold = 1
    audio = r.listen(source)
    with open("recording1.wav", "wb") as f:
        f.write(audio.get_wav_data())
    #i want the code to write {audio} to a wav file called "recording1.wav"
# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
openai.api_key = "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"
audio_file= open("D:\\drive D\\Downloads\\Python apps\\copilot\\GoldWriter\\recording1.wav", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)
print(transcript)