import base64
import json
from datetime import datetime

import uvicorn
from anthropic import Anthropic
from fastapi import FastAPI, Form, Response, WebSocket, WebSocketDisconnect
from pymongo.mongo_client import MongoClient
from twilio.rest import Client
from twilio.twiml.voice_response import Gather, Say, Start, Stop, VoiceResponse

from constants import (
    ANTHROPIC_API_KEY,
    EMBEDDINGS_FIELD,
    MONGO_URI,
    WEBHOOK_DOMAIN,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_PHONE_NUMBER,
    VECTOR_SEARCH_INDEX,
)
from embeddings import vector_search
from prompts import INTRODUCTION, get_language, get_speech, send_message
from voice import transcribe_audio

DEFAULT_VOICE = "Google.en-GB-Wavenet-B"  # British English

app = FastAPI()
anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)
mongo = MongoClient(MONGO_URI)
embeddings_database = mongo.embeddings_database
collection = embeddings_database.embeddings
twilio = Client(username=TWILIO_ACCOUNT_SID, password=TWILIO_AUTH_TOKEN)
database = {}


def hms():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


@app.post("/call")
async def call(CallSid: str = Form(), From: str = Form()):
    print(f"[{hms()}] Incoming call from {From}")
    database[CallSid] = {
        "booking": {},
        "from": From,
        "message_history": [],
        "transcripts": [],
    }
    response = VoiceResponse()

    greeting = INTRODUCTION
    print(f"[{hms()}] Greeting the caller with: '{greeting}'")
    say = Say("", language="en-GB", voice=DEFAULT_VOICE)
    say.prosody(greeting, rate="135%")
    response.append(say)

    response.redirect(f"https://{WEBHOOK_DOMAIN}/record", method="POST")
    return Response(content=str(response), media_type="application/xml")


@app.post("/record")
async def record(CallSid: str = Form()):
    print(f"[{hms()}] Recording")
    response = VoiceResponse()
    database[CallSid]["transcripts"].append("")

    start = Start()
    start.stream(url=f"wss://{WEBHOOK_DOMAIN}/transcribe", name=CallSid)
    response.append(start)
    gather = Gather(
        action="/stop",
        actionOnEmptyResult=True,
        input="speech",
        speechTimeout="auto",
        profanityFilter=False,
        transcribe=False,
    )
    response.append(gather)
    return Response(content=str(response), media_type="application/xml")


@app.post("/stop")
async def stop(CallSid: str = Form()):
    print(f"[{hms()}] Stopping recording")
    response = VoiceResponse()
    stop = Stop()
    stop.stream(name=CallSid)
    response.append(stop)
    response.redirect(f"https://{WEBHOOK_DOMAIN}/respond", method="POST")
    return Response(content=str(response), media_type="application/xml")


@app.post("/respond")
async def respond(CallSid: str = Form()):
    transcript = database[CallSid]["transcripts"][-1]

    print(f"[{hms()}] Waiting for transcript")
    while not transcript:
        continue

    print(f"[{hms()}] Obtaining the most relevant documents from the database")
    documents = vector_search(
        query=transcript,
        collection=collection,
        index=VECTOR_SEARCH_INDEX,
        embeddings_field=EMBEDDINGS_FIELD,
    )

    response = VoiceResponse()

    print(f"[{hms()}] Responding to message: '{transcript}'")

    print(f"[{hms()}] Calling claude-instant-1")
    completion = send_message(
        message=transcript,
        message_history=database[CallSid]["message_history"],
        anthropic=anthropic,
        documents=documents,
    )
    print(f"[{hms()}] Received the claude-instant-1 completion: '{completion}'")

    speech = get_speech(completion)
    voice = get_language(completion)["voice"]
    say = Say("", voice=voice)
    say.prosody(speech, rate="135%")
    response.append(say)

    if "<hangup/>" in completion:
        print(f"[{hms()}] Hanging up")
        response.hangup()

    response.redirect(f"https://{WEBHOOK_DOMAIN}/record", method="POST")
    return Response(content=str(response), media_type="application/xml")


# Twilio's voice transcription is inaccurate and requires the language to be
# specified in advance. Waiting for Twilio to provide recordings is also slow.
# By listening in to the call, using websockets, and transcribing it ourselves,
# we can support multilingual transcriptions, improve accuracy and reduce the
# latency by ~5 seconds.
@app.websocket("/transcribe")
async def transcribe(websocket: WebSocket) -> None:
    """Fast, accurate multilingual audio transcription over websockets."""
    await websocket.accept()

    chunks = []
    call_sid = None
    while True:
        try:
            message = await websocket.receive_text()
            data = json.loads(message)

            if data["event"] == "start":
                call_sid = data["start"]["callSid"]

            elif data["event"] == "media":
                payload = data["media"]["payload"]
                chunks.append(base64.b64decode(payload))

            elif data["event"] == "stop" or data["event"] == "closed":
                break

        except WebSocketDisconnect:
            break

    database[call_sid]["transcripts"][-1] = transcribe_audio(chunks)


# TODO: Send the transcript to a lawyer for review
@app.post("/status")
async def status(CallSid: str = Form(), CallStatus: str = Form()) -> None:
    if CallStatus == "completed":
        twilio.messages.create(
            body="Thank you for calling! If you need further assistance, please call or text us any time.",
            from_=TWILIO_PHONE_NUMBER,
            to=database[CallSid]["from"],
        )


# TODO: handle SMS

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
