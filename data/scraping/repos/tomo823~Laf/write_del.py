#download URL from YouTube and convert it into text files
#mp3 files which was downloaded will be deleted finally
#DONT push this file to GitHub beacause of API key

import openai, os, mimetypes, logging, sys
from dotenv import load_dotenv
from yt_dlp import YoutubeDL


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")

def youtube(URLS):
    
    ydl_opts = {
        'format': 'mp3/bestaudio/best',
        'ignoreerrors': True,

        'postprocessors': [{  
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            
        }]
    }

    with YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download(URLS)
        
    for files in os.listdir("."):
        if mimetypes.guess_type(files)[0] == "audio/mpeg":
            file_name = files.split(".mp3")
            video_name.append(file_name[0])

def text(video):
    f = open(f"{video}.mp3", "rb")
    transcript = openai.Audio.transcribe("whisper-1", f)

    with open(f"{video}.txt", "w", encoding="UTF-8") as file:
        file.write(transcript["text"])
        file.close()
    os.remove(f"{video}.mp3")


if __name__ == "__main__":
    url = input("URL: ")
    video_name = []
    youtube(url)
    for video in video_name:
        text(video)

