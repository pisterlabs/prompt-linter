import requests
import json
from pathlib import Path
# url = 'http://localhost:5000/api'
url = 'http://127.0.0.1:5000/api' #local
""""
发送一个这样的包到后端来：{type": int,
        "content": "string",
        "is_audio_input": False,
        "audio_input": "path",
        "gen_img": False}

        

b返回一个这样的包：{{
  "image": null,
  "result": "response",
  "type": int
}}
"""


# import openai
# audio_file= open("/path/to/file/audio.mp3", "rb")
# transcript = openai.Audio.transcribe("whisper-1", audio_file)
# voice_google_frontend_engineer = open("openAI-api/framework/voices/Google_frontend_engineer.mp3", "rb")
voice_path = Path.cwd()/ "openAI-api"/"framework"/"voices_test"/"Google_frontend_engineer.mp3"
img_save_path = Path.cwd()/ "openAI-api"/"framework"/"images"/"image_test"
img_name= "output.png" #x.__dict__()['result'][:5]
# "./framework/voices/Google_frontend_engineer.mp3"
voice_google_frontend_engineer = open(voice_path, "rb")

select = input("Please select a function: \n 0: test \n 1: job finding \n 2: instruction \n 3: predict future \n 4: voice + job finding\n 5: generate image\n")

if select == "0":
    myobj = {
        "type": 0,
        "content": "test",
        "is_audio_input": False,
        "audio_input": "",
        "gen_img": False
    }
    x = requests.post(url, json = myobj)
    print(x.text)
    exit()

if select == "1":
    user_describe = input("Please describe yourself and the job you are longing for, I will calculate the possibility based on your current position: \n")
    myobj = {
        "type": 1,
        "content": user_describe,
        "is_audio_input": False,
        "audio_input": "",
        "gen_img": False
    }
    x = requests.post(url, json = myobj)
    print(x.text)
    exit()

elif select == "2":
    user_describe = input("Please describe yourself and the job you are longing for, I will show you instructions that you will be better fit this job: \n")
    myobj = {
        "type": 2,
        "content": user_describe,
        "is_audio_input": False,
        "audio_input": "",
        "gen_img": False
    }
    x = requests.post(url, json = myobj)
    print(x.text.replace("\\n", "\n"))
    exit()

elif select == "3":
    user_describe = input("Please describe yourself , I will show you the possible future of yourself: \n")
    myobj = {
        "type": 3,
        "content": user_describe,
        "is_audio_input": False,
        "audio_input": "",
        "gen_img": False
    }
    x = requests.post(url, json = myobj)
    print(x.text.replace("\\n", "\n"))
    exit()
elif select == "4":
    myobj = {
        "type": 1,
        "content": "",
        "is_audio_input": True,
        "audio_input": "",
        "gen_img": False
    }

    voice = {'file': voice_google_frontend_engineer}
    response = requests.post(url+'/uploadMP3', files=voice)
    x = requests.post(url, json = myobj)
    print(x.text)
    exit()
elif select == "5":
    myobj = {
        "type": 1,
        "content": "And draw a scenery picture!",
        "is_audio_input": False,
        "audio_input": "",
        "gen_img": True
    }
    x = requests.post(url, json = myobj)
    print(x.text)
    image_url = url+"/get_image"
    response = requests.get(image_url)
    if response.status_code == 200:
        # 以二进制写入的方式打开一个文件
        with open(img_save_path/img_name, "wb") as file:
            # 将图片内容写入文件中
            file.write(response.content)
    exit()
else: 
    print("Invalid input, please try again")

print("-----------------------------------")
print("The above result are just for a reference. No law enforcement.\n")
print("-----------------------------------")
