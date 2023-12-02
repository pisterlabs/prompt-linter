import modal
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from pydantic import BaseModel
import shelve
import subprocess
from starlette.responses import FileResponse
import json


stub = modal.Stub("tt", image=modal.Image.debian_slim().pip_install("google-cloud-vision", "opencv-python-headless", "python-multipart", "openai~=0.27.0", "elevenlabslib", "sounddevice", "pydub", "requires.io", "soundfile", "wave", "firebase-admin").apt_install("libportaudio2"))

class Item(BaseModel):
    text: str

volume = modal.SharedVolume()
context = []

@stub.webhook(shared_volumes={"/root/cache": volume}, method="POST", secrets=[modal.Secret.from_name("my-openai-secret"), modal.Secret.from_name("my-firebase-secret")])
def tt(item: Item):
    import openai
    from elevenlabslib import ElevenLabsUser
    from elevenlabslib.helpers import save_bytes_to_path
    import wave
    import firebase_admin
    from firebase_admin import credentials, storage
    import requests
    print(item.text) 
    vn = int(item.text[0])
    text = item.text[1:]
    context.append({"role": "user", "content": text})
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You speak like a young child and you are very bubbly and silly."}, 
                      {"role": "user", "content": "My son is going to talk to you soon, and he really wants a friend that is going to speak to him in an excited and bubbly way. You should respond as if you are a young child like my son. You can relax the rules of grammar to meet this requirement. Acknowledge my request in the way I have asked you to. After you acknowledge, my son will talk to you."}, 
                      {"role": "assistant", "content": "Okay! I wanna talk to your son now, I'm so excited to be friends with him!"}] + context
        )
    response = response["choices"][0]["message"]["content"]
    print(response)
    context.append({"role": "assistant", "content": response})
    print(len(context))
    voices = ["Rachel", "Josh", "Antoni", "Domi", "Elli"]
    user = ElevenLabsUser("2be25268d0c50becf9bf5f10645f50fb") #fill in your api key as a string
    voice = user.get_voices_by_name(voices[vn])[0]  #fill in the name of the voice you want to use. ex: "Rachel"
    result = voice.generate_audio_bytes(response) #fill in what you want the ai to say as a string
    save_bytes_to_path("/root/cache/result.wav",result)
    print(os.listdir())
    print(os.getcwd())

    # return FileResponse("result.wav", media_type='audio/wav', filename='result.wav')
    service_account_info = json.loads(os.environ["FIREBASE_KEY"])
    print(os.listdir())
    print(os.listdir("/root/cache"))
    cred = credentials.Certificate(service_account_info)
    print(f"{len(context)} hello!")
    if len(context) == 2:
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'pettalk-376101.appspot.com'
        })
    

    # Get a reference to the storage service
    bucket = storage.bucket()

    # Create a storage reference to the .wav file
    x = len(context)
    file_ref = bucket.blob(f'audio/result{x}.wav')
    print(os.listdir("/root/cache"))
    # Upload the file to Firebase Storage
    with open('/root/cache/result.wav', 'rb') as file:
        file_ref.upload_from_file(file)
    print(os.listdir("/root/cache"))
    # Get the download URL for the file
    file_ref.make_public()
    url = file_ref.public_url # URL expires in 5 minutes
    print('File URL:', url)
    os.remove("/root/cache/result.wav")
    print(os.listdir("/root/cache"))
    return {"url": url}



