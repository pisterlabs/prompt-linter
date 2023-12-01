from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import openai
import os

openai.api_key = os.environ.get("OPENAI_KEY")

content = """
You are grading a student's response. You will return JSON without any new lines that looks like this:
"{
  accuracy: int;
  clarity: int;
  depth: int;
  overallScore: int;
  answer: string;
  feedback: string;
}". 
Your output should be able to be parsed by a JSON.parse() function.

The accuracy field is how accurate the student’s response is out of 100.
The clarity field is how clear the student’s response is out of 100.
The depth field grades the student’s depth out of 100.
The overallScore field grades the student’s overall response out of 100.
The answer field is an extensive, thorough answer to the prompt.
The feedback field is your written feedback to the student’s response, which should be very extensive and explain how the student can improve.

Here is the prompt: 
"""

app = FastAPI()

origins = ["https://qg-admin.vercel.app/", "https://www.quantguide.io/", "https://quantguide.io/", "https://quant-guide-app-git-dev-quantguide.vercel.app/", "https://www.quant-guide-app-git-dev-quantguide.vercel.app/"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"])

@app.get("/")
def test():
    return {"message": "quantguide.io"}


@app.post("/ai")
def ai(file: UploadFile, prompt: str = Form(...)):
    try:
        contents = file.file.read()
        with open(file.filename, "wb") as f:
            f.write(contents)

            transcription = get_transcription(file)

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": content + prompt,
                },
                {
                    "role": "user",
                    "content": transcription,
                },
            ],
            temperature=0.8,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return {"feedback": response["choices"][0]["message"]["content"], "transcript": transcription}
    
    except Exception as e:
        return {"message": e}
    finally:
        file.file.close()
        if os.path.exists(file.filename):
            os.remove(file.filename)


def get_transcription(file):
    try:
        text = ""
        with open(file.filename, "rb") as f:
            text = openai.Audio.transcribe("whisper-1", f)["text"]
        return text
    finally:
        if os.path.exists(file.filename):
            os.remove(file.filename)


@app.post("/transcribe")
def transcribe(file: UploadFile):
    try:
        contents = file.file.read()
        with open(file.filename, "wb") as f:
            f.write(contents)
        return get_transcription(file)
    except Exception as e:
        return {"message": e}
    finally:
        file.file.close()
        if os.path.exists(file.filename):
            os.remove(file.filename)


@app.post("/ai-text")
def ai_text(text: str = Form(...), prompt: str = Form(...)):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": content + prompt,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        temperature=0.8,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    return {"feedback": response["choices"][0]["message"]["content"], "transcript": text}
