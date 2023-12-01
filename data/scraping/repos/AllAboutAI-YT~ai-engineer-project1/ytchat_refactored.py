from pytube import YouTube
from moviepy.editor import AudioFileClip
import openai
import requests
import os

# Set the API key from an environment variable for security
api_key = 'YOUR API KEY'


# Function to download a YouTube video
def download_youtube_video(video_url):
    try:
        yt = YouTube(video_url)
        video_stream = yt.streams.filter(file_extension='mp4').first()
        if video_stream:
            video_path = video_stream.download()  # This will use the default download path and filename
            return video_path, None  # No error
        else:
            return None, "No available video streams match the criteria."
    except Exception as e:
        return None, f"An error occurred: {e}"

# Function to convert MP4 to MP3
def convert_mp4_to_mp3(mp4_file_path, mp3_file_path):
    try:
        video_clip = AudioFileClip(mp4_file_path)
        video_clip.write_audiofile(mp3_file_path)
        video_clip.close()
        return mp3_file_path, None  # Indicate that there was no error
    except Exception as e:
        return None, f"An error occurred during conversion: {e}"

# Function to transcribe audio to text
def transcribe_audio_to_text(audio_file_path):
    """
    Transcribe the given audio file to text using OpenAI's Whisper API.

    :param audio_file_path: str - Path to the audio file to transcribe.
    :return: Tuple (transcribed_text, error_message)
    """
    headers = {
        'Authorization': f'Bearer {api_key}'
    }

    try:
        with open(audio_file_path, 'rb') as audio_file:
            files = {
                'file': audio_file,
                'model': (None, 'whisper-1'),
                # If you know the language, uncomment the next line
                # 'language': (None, 'en')  # Replace 'en' with the appropriate language code if needed
            }
            response = requests.post('https://api.openai.com/v1/audio/transcriptions', headers=headers, files=files)

        if response.status_code == 200:
            # Extract the transcription text
            text = response.json()["text"]
            return text, None
        else:
            # Return None and the error message
            return None, f"An error occurred: {response.status_code} - {response.text}"
    except Exception as e:
        # Catch any exceptions and return None and the error message
        return None, f"An exception occurred: {e}"

# Function to create chatbot context from transcribed text
def create_chatbot_context(transcribed_text):
    return [{"role": "system", "content": transcribed_text}]

# Function to chat with the bot
def chat_with_bot(conversation, user_message, api_key):
    openai.api_key = api_key
    try:
        conversation.append({"role": "user", "content": user_message})
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",  # replace with the correct model ID if different
            messages=conversation
        )
        assistant_message = response.choices[0].message["content"]
        conversation.append({"role": "assistant", "content": assistant_message})
        return assistant_message, None
    except Exception as e:
        return None, f"An error occurred: {e}"




