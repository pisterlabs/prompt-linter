#Python 3.9
#%% Create a function to download the audio file from the url
import os
def download_audio(url, oname):
    cmd = f'yt-dlp -x --audio-format mp3 -o "{oname}" {url}'
    os.system(cmd)

#%% Import OpenAI library
from openai import OpenAI
#%% Connect to OpenAI
client = OpenAI()
#%% Create a function that points to a file and returns a text with the text categorized as text
def get_text(af):
    audio_file = open(af, "rb")
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="text")
    return transcript
#%% Create a function that writes the text to a .txt file
def write_text(text, name):
    text_file = open(name, "w")
    text_file.write(text)
    text_file.close()

#%% Create a main function that prompts the user for a url, and then downloads the audio file, transcribes it, and writes it to a .txt file
def main():
    url = input("Enter the url of the video: ")
    oname = input("Enter the name of the output file without any extentions: ")
    media_file = oname + ".mp3"
    text_file = oname + ".txt"
    download_audio(url, media_file)
    text = get_text(media_file)
    write_text(text, text_file)
    print("Done!")
#%% Run the main function
main()
