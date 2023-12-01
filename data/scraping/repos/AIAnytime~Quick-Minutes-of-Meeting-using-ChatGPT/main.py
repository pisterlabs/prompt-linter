import whisper
from fastapi import FastAPI, File, UploadFile 
from fastapi.responses import RedirectResponse, JSONResponse
import aiofiles
import subprocess
import openai
import os
import sys
import json

app = FastAPI()

openai.api_key = ""

model = whisper.load_model("base")

def video_to_audio(video_file):
    audio_file = "input_audio.mp3"
    subprocess.call(["ffmpeg", "-y", "-i", video_file, audio_file], 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    return audio_file

def audio_to_transcript(audio_file):
    result = model.transcribe(audio_file)
    transcript = result["text"]
    print(transcript)
    return transcript

def MoM_generation(prompt):
    response = openai.Completion.create(model="text-davinci-003",
                                        prompt= "Can you generate the Minute of Meeting in form of bullet points for the below transcript?\n"+prompt, 
                                        temperature=0.7, 
                                        max_tokens=256, 
                                        top_p=1,
                                        frequency_penalty=0, 
                                        presence_penalty=0)
    return response['choices'][0]['text']

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@app.post("/upload_video")
async def upload_video(file: UploadFile=File(...)):
    filename = file.filename
    async with aiofiles.open(filename, mode='wb') as f:
        await f.write(await file.read())
    
    audio_file = video_to_audio(filename)
    transcript = audio_to_transcript(audio_file)    
    final_result = MoM_generation(transcript)
    response_body = final_result.replace('\n', ' ')
    response_dict = {"response": response_body}
    json_result = json.dumps(response_dict)
    return json.loads(json_result)
    