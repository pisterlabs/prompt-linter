# Description: This example shows how to convert speech to text using the OpenAI API.
# Source: https://platform.openai.com/docs/guides/speech-to-text

from openai import OpenAI
client = OpenAI()

mp3_file = "/Users/mmuyakwa/scripts/AI/OpenAI-Playground/text-to-speech/speech.mp3"

audio_file = open(mp3_file, "rb") # Accepted formats: mp3, mp4, mpeg, mpga, m4a, wav, and webm
transcript = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file, 
  response_format="text"
)

# Show the transcript
print(transcript)