from openai import OpenAI
client = OpenAI()

audio_file = open("./output2.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    prompt="This is about DALL-E 3."
)
print(transcript.text)
