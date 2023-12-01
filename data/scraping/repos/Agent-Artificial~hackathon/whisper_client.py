from openai import OpenAI
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

local_whisper_url = os.getenv("local_whisper_url")
if (local_whisper_url): print("local model: ", local_whisper_url)

# Define the base API URL and the audio file path at the top
api_base = local_whisper_url
audio_file_path  = 'jfk.wav'  # The audio file is now a variable at the top

# Set the OpenAI API base URL and key
client = OpenAI(
   base_url = api_base,
   api_key = "NULL",
)

def transcribe_audio(audio_file_path):
    try:
        with open(audio_file_path, 'rb') as audio_file:
            print(f"Transcribing audio file: {audio_file_path}")
            #print(f"audio_file: ", audio_file)
            # Transcribe the audio file
            response = client.audio.transcriptions.create(
                         model="whisper-1",
                         file=audio_file,
                         response_format="text"
                       )

            # Extract the transcription text
            transcription = response
            #print(f"Transcription successful: {transcription}")
            return transcription
    except Exception as e:
        print(f"An error occurred during transcription: {e}")

# Example usage:
if __name__ == "__main__":
    try:
        transcription = transcribe_audio(audio_file_path)
        if transcription:
            print(f"Transcription: {transcription.text}")
        else:
            print("No transcription was returned.")
    except FileNotFoundError:
        print(f"The file {audio_file_path} was not found.")
    except Exception as e:
        print(f"An error occurred during the process: {e}")