import sys
from openai import OpenAI
import pygame

# Check if an argument was passed
if len(sys.argv) < 2:
    print("Please provide a text input.")
    sys.exit()

input_text = sys.argv[1]

client = OpenAI()

response = client.audio.speech.create(
    model="tts-1",
    voice="onyx",
    input=input_text,
)

# Save the audio file
speech_file_path = "output.mp3"
response.stream_to_file(speech_file_path)

# Initialize pygame mixer
pygame.mixer.init()

# Load the audio file
pygame.mixer.music.load(speech_file_path)

# Play the audio
pygame.mixer.music.play()

# Wait for playback to finish
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)


