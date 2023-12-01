# Description: This script generates speech from text with the text-to-speech model from OpenAI.
# Source: https://platform.openai.com/docs/guides/text-to-speech

from pathlib import Path
from openai import OpenAI
client = OpenAI()

# Create a file path for the generated speech
speech_file_path = Path(__file__).parent / "speech.mp3"

# Generate speech with the text-to-speech model from OpenAI
response = client.audio.speech.create(
  model="tts-1",
  voice="nova", # Available Voices: alloy, echo, fable, onyx, nova, and shimmer
  input="Funktioniert die Sprachausgabe auch mit deutschen Texten?"
)

# Save the generated speech to the file path
response.stream_to_file(speech_file_path)
