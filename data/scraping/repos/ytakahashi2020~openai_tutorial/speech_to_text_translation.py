from openai import OpenAI
client = OpenAI()

audio_file = open("./output3.mp3", "rb")
translation = client.audio.translations.create(
    model="whisper-1",
    file=audio_file
)
print(translation.text)
