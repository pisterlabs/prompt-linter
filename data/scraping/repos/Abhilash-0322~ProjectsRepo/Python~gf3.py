import openai
from gtts import gTTS
import os
import tempfile
import pygame

openai.api_key = "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"

# Initialize pygame mixer for audio playback
pygame.mixer.init()

while True:
    user_input = input("You: ")

    # Generate a response from OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
             {
                "role": "system",
                "content": "embrace the role of an extremely caring girlfriend"
            },
            {
                "role": "user",
                "content": "You mean the world to me, and your care and love brighten my days."
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=0.2,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Access and print the generated response
    generated_text = response["choices"][0]["message"]["content"]
    print(f"AI (Text): {generated_text}")
    

    # Generate TTS audio from the AI's response
    tts = gTTS(generated_text)
    
    # Create a temporary audio file
    temp_dir = os.path.join(os.getcwd(), "temp_audio")

# Ensure the temporary directory exists
    os.makedirs(temp_dir, exist_ok=True)

# Create a temporary audio file
    temp_audio_file = tempfile.NamedTemporaryFile(delete=True, suffix=".mp3", dir=temp_dir)
    temp_audio_file_name = temp_audio_file.name
    
    # Save TTS audio to the temporary file
    tts.save(temp_audio_file_name)

    # Play the TTS audio
    pygame.mixer.music.load(temp_audio_file_name)
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        pass

    # Close and clean up the temporary audio file
    temp_audio_file.close()
