import openai

client = openai.OpenAI()

audio_file = open("sample-0.mp3", "rb")
transcript = client.audio .transcriptions.create(
    model="whisper-1",
    file=audio_file
)
print(transcript.text)