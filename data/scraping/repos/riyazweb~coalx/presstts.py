import os
 
from moviepy.video.VideoClip import TextClip
from moviepy.video.tools.subtitles import SubtitlesClip
 
from bing_image_downloader import downloader
from moviepy.editor import *
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.fx.all import volumex

from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import ColorClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.fx.all import resize
import moviepy.editor as mp
 
import os
from whisper.utils import write_srt
import whisper
import openai
import re
 
import os
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

url = "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3"

response = requests.get(url)

if response.status_code == 200:
    root = ET.fromstring(response.content)

    first_link_element = root.find(".//item/link")
    
    if first_link_element is not None:
        first_link = first_link_element.text

        link_response = requests.get(first_link)

        if link_response.status_code == 200:
            soup = BeautifulSoup(link_response.text, 'html.parser')
            
            text_content = soup.get_text()

            content = ' '.join(text_content.split())

            with open("content.txt", "w", encoding="utf-8") as file:
                file.write(content)
            print(f"Cleaned content saved as 'content.txt'")
        else:
            print("Failed to fetch the content from the link.")
    else:
        print("No link found in the XML.")
else:
    print("Failed to fetch the XML from the URL.")
openai.api_key = 'api key here' 




def main():
   

    print("Welcome to  Chat!")

    while True:
        
            cop = input("You: ")

            with open('content.txt', 'r') as file:
                content = file.read()

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "user", "content": f"summarize this text in 70 words: {content}"}
                ]
            )

            OP = response['choices'][0]['message']['content']
            OP = OP.replace('"', '')

            print(OP)
            
            from feautures.custom_voice import speak
          
            top = input("images:")
            option = 1
         
            fish = top
      
            limit = 7


            
            cop = f"{OP}"
     #for clean pure voice
  # from elevenlabs import generate, play, set_api_key, save

     #       voice = generate(
      #          text=f"{cop}",
       #         voice="Bella",
        #        model="eleven_multilingual_v2"
         #   )
          #  save(voice,'data.mp3')

            speak(cop)

     
            VIDEO_WIDTH = 480
            VIDEO_HEIGHT = 854
         


            MUSIC_PATH = "data.mp3"
            from pygoogle_image import image as pi

            import PIL

            IMAGE_PATHS = []
            if option == 1:
                for element in top.split(','):

                    pi.download(keywords=f'{element}', limit=limit)

                    dog = element.replace(" ", "_")
                    folder_path = f"images/{dog}/"
                    if not os.path.exists(folder_path):
                        continue
                    for f in os.listdir(folder_path):
                        if f.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                            image_path = os.path.join(folder_path, f)
                            with PIL.Image.open(image_path) as img:
                                width, height = img.size
                                if width > 80 and height > 36:
                                    IMAGE_PATHS.append(image_path)

            elif option == 2:
                for element in top.split(','):

                    downloader.download(f"{element}", limit=limit, output_dir="images",
                                        adult_filter_off=True, force_replace=False)

                    folder_path = f"images/{element}/"
                    if not os.path.exists(folder_path):
                        continue
                    image_paths = [os.path.join(folder_path, f) for f in os.listdir(
                        folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                    IMAGE_PATHS += image_paths
            else:
                print("Invalid option entered. Please enter either 1 or 2.")

            num_images = len(IMAGE_PATHS)
            audio_clip = AudioFileClip(MUSIC_PATH)
            audio_duration = audio_clip.duration
            IMAGE_DURATION = audio_duration / num_images

            video_clips = []
            for image_path in IMAGE_PATHS:
                image_clip = ImageClip(image_path)
                new_height = int(
                    VIDEO_WIDTH / image_clip.w * image_clip.h)
                image_clip = image_clip.resize(
                    (VIDEO_WIDTH, new_height))
                image_clip = image_clip.set_position(
                    ("center", "center"))
                image_clip = image_clip.set_duration(IMAGE_DURATION)

                bg_clip = ColorClip(
                    (VIDEO_WIDTH, VIDEO_HEIGHT), color=(0, 0, 0))
                bg_clip = bg_clip.set_duration(IMAGE_DURATION)

                video_clip = CompositeVideoClip([bg_clip, image_clip])

                video_clips.append(video_clip)


            audio_clip = AudioFileClip(MUSIC_PATH)
            audio_duration = audio_clip.duration
            final_clip = concatenate_videoclips(
                video_clips, method="compose", bg_color=(0, 0, 0))
            final_clip = final_clip.set_duration(
                audio_duration).loop(duration=audio_duration)

            final_clip = final_clip.set_audio(
                audio_clip.set_duration(final_clip.duration))

            filename = f"{fish}.mp4"

            if os.path.isfile(filename):
                basename, extension = os.path.splitext(filename)
                i = 1
                while os.path.isfile(f"{basename}_{i}{extension}"):
                    i += 1
                filename = f"{basename}_{i}{extension}"

            final_clip.write_videofile(filename, fps=30)


            FONT = "Muroslant.otf"
            input_path = f"{fish}.mp4"
            print("Transcribing audio...")
            model = whisper.load_model("base")
            result = model.transcribe(input_path, verbose=False)

            subtitle_path = os.path.splitext(input_path)[0] + ".srt"
            with open(subtitle_path, "w", encoding="utf-8") as srt_file:
                write_srt(result["segments"], file=srt_file)

            print("Generating subtitles...")
            orig_video = VideoFileClip(input_path)

            def generator(txt): return TextClip(txt,
                                                font=FONT if FONT else "Courier",
                                                fontsize=38,
                                                color='white',
                                                size=orig_video.size,
                                                method='caption',
                                                align='center',)
            subs = SubtitlesClip(subtitle_path, generator)

            print("Compositing final video...")
            final = CompositeVideoClip(
                [orig_video, subs.set_position('center', 'middle')])
            final_path = os.path.splitext(input_path)[0] + "_final.mp4"
            final.write_videofile(final_path, fps=orig_video.fps)
            file_names = [f"{filename}"]
            for file_name in file_names:
                if os.path.exists(file_name):
                    os.remove(file_name)
                    print(f"File '{file_name}' deleted successfully.")
                else:
                    print(f"File '{file_name}' does not exist.")

        

if __name__ == "__main__":
    main()
