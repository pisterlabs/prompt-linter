from pytube import YouTube
from moviepy.editor import AudioFileClip
import openai
import requests

# Load your OpenAI API key from an environment variable or direct assignment
openai.api_key = 'YOUR API KEY'
api_key = openai.api_key

# Define the URL variable for the YouTube video you want to download.
def download_youtube_video(video_url):
    """
    Downloads a YouTube video from the provided URL in mp4 format 
    with the filename 'temp_video.mp4'.
    
    :param video_url: str - The YouTube video URL to download.
    """
    try:
        # Create a YouTube object with the URL
        yt = YouTube(video_url)
        
        # Select the stream: you might want to add additional filters here
        # for resolution or other stream properties.
        video_stream = yt.streams.filter(file_extension='mp4').first()

        # Ensure there is a stream available to download
        if video_stream:
            # Download the video
            video_stream.download(filename='temp_video.mp4')
            print("Download completed successfully!")
        else:
            print("No available video streams match the criteria.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

def convert_mp4_to_mp3(mp4_file_path, mp3_file_path):
    """
    Converts an MP4 video file into an MP3 audio file.

    :param mp4_file_path: str - Path to the input MP4 video file.
    :param mp3_file_path: str - Path for the output MP3 audio file.
    """
    try:
        # Load the MP4 file
        video_clip = AudioFileClip(mp4_file_path)
        
        # Write the audio to an MP3 file
        video_clip.write_audiofile(mp3_file_path)
        
        # Close the video clip to release the resources
        video_clip.close()
        
        print(f"MP4 to MP3 conversion completed successfully! MP3 saved as: {mp3_file_path}")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")
     

def transcribe_audio_to_text(api_key, audio_file_path, output_text_file):
    """
    Transcribe the given audio file to text using OpenAI's Whisper API, print and save to a text file.

    :param api_key: str - Your OpenAI API key.
    :param audio_file_path: str - Path to the audio file to transcribe.
    :param output_text_file: str - Path to the output text file.
    """
    headers = {
        'Authorization': f'Bearer {api_key}'
    }

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
        
        # Print the transcribed text
        print(text)
        
        # Write the transcription to a text file
        with open(output_text_file, 'w', encoding='utf-8') as text_file:
            text_file.write(text)
        print(f"Transcription saved to {output_text_file}")
    else:
        print(f"An error occurred: {response.status_code} - {response.text}")
       

def create_chatbot_context(transcribed_text_file):
    # Read the transcribed text and return it as initial context for the chat
    with open(transcribed_text_file, 'r', encoding='utf-8') as file:
        transcribed_text = file.read()
    return [{"role": "system", "content": transcribed_text}]

def chat_with_bot(transcribed_text_file):
    # Initialize the conversation list with the transcribed text
    conversation = create_chatbot_context(transcribed_text_file)

    # Main loop for the chat
    while True:
        # Get user input
        user_message = input("You: ")
        if user_message.lower() == 'quit':
            print("Exiting the chat.")
            break

        # Append the user message to the conversation
        conversation.append({"role": "user", "content": user_message})

        # Call the OpenAI API to get a response
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",  # replace with the correct model ID if different
            messages=conversation
        )

        # Extract the assistant's message from the response
        assistant_message = response.choices[0].message["content"]
        print(f"Assistant: {assistant_message}")

        # Append the assistant message to the conversation
        conversation.append({"role": "assistant", "content": assistant_message})
       
            


