import io
import os
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY"),
  base_url=os.getenv("OPENAI_API_BASE"),
)

def stream_and_play(text):
  response = client.audio.speech.create(
    model="tts-1",
    voice="shimmer",#"alloy",
    input=text,
  )

  # Convert the binary response content to a byte stream
  byte_stream = io.BytesIO(response.content)

  # Read the audio data from the byte stream
  audio = AudioSegment.from_file(byte_stream, format="mp3")

  # Play the audio
  play(audio)


if __name__ == "__main__":
  while True:
    text = input("Enter text: ")
    if text == "quit":
      break
    stream_and_play(text)