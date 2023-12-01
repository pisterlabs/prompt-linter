from openai import OpenAI

with open("data/chatgpt/openai.env") as f:
    key = f.read()

with open("data/chatgpt/python.txt") as f:
    text = f.read()

print(text)

client = OpenAI(api_key=key)

response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input=text[:4096]
)

response.stream_to_file("data/chatgpt/out.mp3")
