from moviepy.editor import *
import openai
import json
import urllib.request
import pixabay.core
import random
from gtts import gTTS
import os
from moviepy.editor import VideoFileClip, CompositeVideoClip
from serpapi import GoogleSearch
import requests
from PIL import Image
from io import BytesIO
from moviepy.editor import *
from moviepy.config import change_settings
import requests
from mutagen.mp3 import MP3


openai.api_key = ""
API_KEY = ""
change_settings({"IMAGEMAGICK_BINARY": r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})

def chatgpt(querytext):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages= [
        {"role": "user", "content": querytext},
    ]
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result


def pixabayy(querypicture):
    BASE_URL = "https://pixabay.com/api/"
    API_KEY = ""

    params = {
        "key": API_KEY,
        "q": querypicture,
        "image_type": "photo",
        "per_page": 5
    }

    response = requests.get(BASE_URL, params=params)

    data = response.json()

    for i, hit in enumerate(data["hits"]):
        image_url = hit["largeImageURL"]
        txtname= querypicture+str(i)+".jpg"
        file_name = txtname
        file_path = os.path.join("D:\\VScode\\TrueStory", file_name)
        response = requests.get(image_url)
        with open(file_path, "wb") as f:
            f.write(response.content)

        dir_path = "D:\\VScode\\TrueStory"

        img = Image.open('D:\\VScode\\TrueStory\\'+file_name)

        img_resized = img.resize((300, 300))
        img_resized = img_resized.convert('RGB')
        img_resized.save('D:\\VScode\\TrueStory\\'+file_name, quality=25)

def serpapii(querypicture2):
    params = {
    "q": querypicture2,
    "tbm": "isch",
    "ijn": "0",  
    "api_key": ""
}

    search = GoogleSearch(params)
    results = search.get_dict()

    try:
        for i, result in enumerate(results['images_results']):
            if i >= 4:  
                break
            url = result['original']
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            txtt= querypicture2+i+".jpg"
            img.save(txtt)  
            image = Image.open(txtt)
            new_size = (300, 300)
            resized_image = image.resize(new_size)

            resized_image.save(txtt)
    except:
        print("On the way!")



def piccombine(querypicture3, timeperpic, numofpic, picch):
    image_files=[]
    for i in range(0, numofpic):
        img = Image.open('D:\\VScode\\TrueStory\\'+(querypicture3+str(i)+".jpg"))
        img= img.resize((300, 300))
        img = img.convert('RGB')
        img.save('D:\\VScode\\TrueStory\\'+(querypicture3+str(i)+".jpg"), quality=25)

        descrimage=querypicture3+str(i)+".jpg"
        image_files.append(descrimage)

    duration = timeperpic

    video_name = 'picoutput.mp4'
    
    image_clips = [ImageClip(image_file).set_duration(duration) for image_file in image_files]

    
    video_clip = concatenate_videoclips(image_clips)
    video_clip.write_videofile(video_name, fps=30)
    if picch.lower()!="manual":
        for i in image_files:
            os.remove(i)
    return video_name


def contentspeech(speechtopic, speech):    
    mytext = speech
    
    language = 'en'
    myobj = gTTS(text=mytext, tld='us', lang=language, slow=False)
    
    myobj.save(speechtopic+".mp3")
    return (speechtopic+".mp3")
    # os.system(speechtopic+".mp3")

def aimusic():  #downloaded a bunch of AI generated music to avoid copyright issues. This process can be baked into the program too but it's an unnecessary step that increases the time taken to produce a short, so I just downloaded it.
    randmusic= random.randint(1,4)
    if randmusic==1:
        song= "aimusic1.mp3"
    if randmusic==2:
        song= "aimusic2.mp3"
    if randmusic==3:
        song= "aimusic3.mp3"
    if randmusic==4:
        song= "aimusic4.mp3"
    if randmusic==5:
        song= "aimusic5.mp3"   

    return song
    

def gamingclip(name, duration):
    n=random.randint(0,1)
    if n==1:
      clip = VideoFileClip("milesclipfinal.mp4")
      dur=random.randint(4, 158-duration)
      clip = clip.subclip(dur, dur+duration)

    else: 
       clip = VideoFileClip("gtavclipfinal.mp4")
       dur=random.randint(6, 263-duration)
       clip = clip.subclip(dur, dur+duration)

    
    output_filename = name
    clip.write_videofile(output_filename,fps=24, codec='libx264', threads=16, verbose= False)



def mainprog():
    while True:
        ch= input("Manual topic entry or automated?: ")
        if "manual" in ch.lower():
            speech= input("Enter content of the video: ")
            picdesc= input("What are the keywords of the pictures to be used?: ")
            topic=speech
            break

        else:    
            topic= input("What do you want the topic of this video to be?: ")
            topicc=topic
            topic= "Write a short description answering the question"+topic+". IF and only if the description is a fact, elaborate on the one fact instead of stating multiple facts. Start it off with intruiging clickbait kinda thought provoking words. Do not include the title or any other unnecessary information. Actually include information about the topic."
            speech= chatgpt(topic)
            speech= speech.replace('"', "")
            speech=speech.replace(",", "")
            speech=speech.replace("\n", " ")
            print("Speech: ", speech)
            chh=input("Is the above good?: ")
            if "yes" in chh.lower():
                picdesc= input("What images should be searched?: ")
                break
            else:
                print("Alright, let's try this again")
                continue

            

    audiofilename = contentspeech(topic[0:10], speech+ " Subscribe for more!")
    audio = MP3(audiofilename)

    audio_info = audio.info    
    length_in_secs = int(audio_info.length)


    picch=input("Pixabay or serpapi or manual picture entry? (Manual entry default pic name must be pic+i): ")        
    if "pixabay" in picch.lower():
        pixabayy(picdesc)
        numofpics=5
    elif "serpapi" in picch.lower():
        serpapii(picdesc)
        numofpics=4
    else:
        numofpics=int(input("How many pictures do you have?: "))
        picdesc="pic"
        


    piccombinename = piccombine(picdesc, length_in_secs/numofpics, numofpics, picch)

    gamingclip("gaem.mp4", length_in_secs+1)
    background_video = VideoFileClip("gaem.mp4")


    
    overlay_video = VideoFileClip(piccombinename).resize(width=500)

    
    overlay_position = ((background_video.w - overlay_video.w) / 2, 30)  

    
    composite_clip = CompositeVideoClip([background_video, overlay_video.set_pos(overlay_position)])
    audio_clip = AudioFileClip(audiofilename)
    song= aimusic()
    bg_music= AudioFileClip(song)
    bg_music= bg_music.volumex(0.7)
    bg_musicf = bg_music.subclip(0,length_in_secs+1)

    concatenated_audio = CompositeAudioClip([bg_musicf, audio_clip])

    composite_clip = composite_clip.set_audio(concatenated_audio)

    l=0
    h=4
    sttart=0
    ennd=2
    for i in range(0, len(speech.split(" ")),4):
                txt = " ".join(speech.split(" ")[l:h])
                l=h
                h+=4
                txt_clip = TextClip(txt.upper(), fontsize=25, color='White', font="Bodoni-MT-Bold", transparent=True)
                txt_clip = txt_clip.set_start(sttart).set_end(ennd)
                sttart=ennd
                ennd+=1.5


                txt_clip = txt_clip.set_pos('center')

                composite_clip = CompositeVideoClip([composite_clip, txt_clip])


    composite_clip.write_videofile('FINALSHORT.mp4', fps=24, codec='mpeg4', threads=16, bitrate='4000k')



mainprog()





