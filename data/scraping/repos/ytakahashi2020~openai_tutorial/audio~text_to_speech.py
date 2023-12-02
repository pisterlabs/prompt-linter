from openai import OpenAI

client = OpenAI()

response = client.audio.speech.create(
    model="tts-1-hd",
    voice="nova",
    input="Hallo, mein Name ist Yuki.Es ist ein schöner Tag heute.",
)

response.stream_to_file("output3.mp3")
