import openai
import requests
import json
from decouple import config
from utils.config import VALLEY_GIRL, SAPHIRA
import pygame

ELEVENLABS_API_KEY = config("ELEVENLABS_API_KEY")
pygame.mixer.init()
# Extract audio file as byte data
def process_audio_file(file):
    with open(file.filename , "wb") as f:
        f.write(file.file.read())
    audio_input = open(file.filename, "rb")

    return audio_input

# Open AI - Whisper 
# Convert user audio to text
def convert_audio_to_text(audio_file):
    try:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        message_text = transcript["text"]
        return message_text
    except Exception as e:
        print("Something wrong when converting audio.")
        return 

# Convert malia text to speech
def convert_text_to_speech(message):
    
    body = {
        "text": message,
        "vocie_settings": {
            "stability": 0.23,
            "similarity_boost": 0.9
        }
    }

    voice_young_american_girl = VALLEY_GIRL

    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json", "accept": "audio/mpeg"}
    endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_young_american_girl}"

    try: 
        response = requests.post(url=endpoint, json=body, headers=headers)
    except Exception:
        print("Request failed when requesting ElevenLabs api")
        return None
    
    if response.status_code == 200 and response.content:
        with open("utils/malia_response.mp3", "wb") as f:
            f.write(response.content)
        
        pygame.mixer.music.load("utils/malia_response.mp3")
        pygame.mixer.music.play()

        return response.content


# # Provide StreamingResponse
# def get_malia_audio(malia_text):
    
#     # Convert AI response to audio
#     malia_audio = convert_text_to_speech(malia_text)
#     if malia_audio is None:
#         return HTTPException(status_code=400, detail="Failed to convert AI message to audio")

#     def iterfile():
#         yield malia_audio

#     return StreamingResponse(iterfile(), media_type="application/octet-stream")
    

# Get the list of voice model data in my ElevenLabs voice library    
def get_voices():
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    endpoint = f"https://api.elevenlabs.io/v1/voices"

    response = requests.get(url=endpoint, headers=headers)
    
    with open("voices.json", "w") as f:
        
        json.dump(response.json(), f , indent=3)
    

if __name__ == '__main__':
    import time
    text = "Jesus crist, can just shut up?"
    convert_text_to_speech(text)
    time.sleep(10)