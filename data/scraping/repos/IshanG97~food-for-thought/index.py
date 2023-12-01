import base64
import json
import pathlib
from datetime import datetime

import requests
import uvicorn
from anthropic import Anthropic
from fastapi import FastAPI, Form, Response, WebSocket, WebSocketDisconnect, Request, HTTPException
from langchain.chat_models import ChatAnthropic
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import (
    TextLoader,
    WebBaseLoader,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import Chroma
from pydantic import BaseModel
from twilio.rest import Client
from twilio.twiml.voice_response import Gather, Say, Start, Stop, VoiceResponse

import constants
from prompts import SYSTEM_MESSAGE, format_snippets, get_booking, get_speech
from voice import transcribe_audio

RESTAURANT = "ChIJ8YR2BUkbdkgRmxJhDIsuy2U"
NUM_SNIPPETS = 3
VOICE = "Google.en-GB-Wavenet-B"

app = FastAPI()
anthropic = Anthropic(api_key=constants.ANTHROPIC_API_KEY)
twilio = Client(username=constants.TWILIO_ACCOUNT_SID, password=constants.TWILIO_AUTH_TOKEN)

calls = {}
model = ChatAnthropic(
    model="claude-instant-1",
    anthropic_api_key=constants.ANTHROPIC_API_KEY,
)
restaurants = {}


def hms():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


@app.post("api/call")
async def call(CallSid: str = Form(), From: str = Form()):
    print(f"[{hms()}] Incoming call from {From}")

    # Initialise call data
    restaurant_name = restaurants[RESTAURANT]["place_details"]["name"]
    calls[CallSid] = {
        "booking": {},
        "from": From,
        "message_history": [],
        "restaurant_name": restaurant_name,
        "transcripts": [],
    }

    # Set up agent
    calls[CallSid]["agent"] = {"index": restaurants[RESTAURANT]["index"]}

    response = VoiceResponse()
    greeting = f"Thank you for calling {restaurant_name}!"
    print(f"[{hms()}] Greeting the caller with: '{greeting}'")
    say = Say("", language="en-GB", voice=VOICE)
    say.prosody(greeting, rate="135%")
    response.append(say)

    response.redirect(f"https://{constants.NGROK_DOMAIN}/record", method="POST")
    return Response(content=str(response), media_type="application/xml")


@app.post("api/record")
async def record(CallSid: str = Form()):
    print(f"[{hms()}] Recording")
    response = VoiceResponse()
    calls[CallSid]["transcripts"].append("")

    start = Start()
    start.stream(url=f"wss://{constants.NGROK_DOMAIN}/transcribe", name=CallSid)
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


@app.post("api/stop")
async def stop(CallSid: str = Form()):
    print(f"[{hms()}] Stopping recording")
    response = VoiceResponse()
    stop = Stop()
    stop.stream(name=CallSid)
    response.append(stop)
    response.redirect(f"https://{constants.NGROK_DOMAIN}/respond", method="POST")
    return Response(content=str(response), media_type="application/xml")


@app.post("api/respond")
async def respond(CallSid: str = Form()):
    transcript = calls[CallSid]["transcripts"][-1]

    print(f"[{hms()}] Waiting for transcript")
    while not transcript:
        continue

    print(f"[{hms()}] Responding to message: '{transcript}'")
    response = VoiceResponse()

    print(f"[{hms()}] Obtaining relevant snippets from the database")
    index = calls[CallSid]["agent"]["index"]
    search = index.similarity_search_with_score(transcript, k=NUM_SNIPPETS)
    snippets = [d[0].page_content for d in search]

    print(f"[{hms()}] Calling claude-instant-1")
    message_history = calls[CallSid]["message_history"]
    message_history.append(HumanMessage(content=transcript))
    messages = [
        SystemMessage(
            content=SYSTEM_MESSAGE.format(
                snippets=format_snippets(snippets),
            )
        )
    ] + message_history
    completion = model(messages).content
    message_history.append(AIMessage(content=completion))
    print(f"[{hms()}] Received the claude-instant-1 completion: '{completion}'")

    say = Say("", voice=VOICE)
    say.prosody(get_speech(completion), rate="135%")
    response.append(say)

    if "<booking>" in completion:
        booking = get_booking(completion)
        if booking:
            print(f"[{hms()}] Booking details: {booking}")
            calls[CallSid]["booking"] = booking

    if "<hangup/>" in completion:
        print(f"[{hms()}] Hanging up")
        response.hangup()

    response.redirect(f"https://{constants.NGROK_DOMAIN}/record", method="POST")
    return Response(content=str(response), media_type="application/xml")


@app.websocket("api/transcribe")
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

    calls[call_sid]["transcripts"][-1] = transcribe_audio(chunks)


BOOKING_MESSAGE = """Your booking for {num_people} people under the name {name}
at {time} has been confirmed. We look forward to seeing you! {restaurant_name}
""".replace(
    "\n", " "
)


@app.post("api/status")
async def status(CallSid: str = Form(), CallStatus: str = Form()) -> None:
    """Sends a booking confirmation via SMS."""
    if CallStatus == "completed":
        if booking := calls[CallSid]["booking"]:
            phone_number = calls[CallSid]["from"]
            print(f"[{hms()}] Sending booking confirmation to {phone_number}")
            twilio.messages.create(
                body=BOOKING_MESSAGE.format(
                    num_people=booking["num_people"],
                    name=booking["name"],
                    time=booking["time"],
                    restaurant_name=calls[CallSid]["restaurant_name"],
                ),
                from_=constants.TWILIO_PHONE_NUMBER,
                to=phone_number,
            )


@app.get("api/save-restaurant-data")
def save_restaurant_data(place_id: str) -> None:
    if place_id in restaurants:
        return

    response = requests.get(
        constants.PLACE_DETAILS_URL,
        params={
            "key": constants.GCP_API_KEY,
            "place_id": place_id,
        },
    )

    if response.status_code == 200:
        place_details = response.json()["result"]

    data_path = f"../data/{place_id}"
    pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)

    # Save reviews
    reviews = place_details["reviews"]
    with open(f"{data_path}/reviews.txt", "w") as file:
        for review in reviews:
            if "text" in review:
                file.write(review["text"] + "\n")
    reviews_data = TextLoader(f"{data_path}/reviews.txt").load()
    reviews_splits = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0,
        separators=[],
    ).split_documents(reviews_data)

    # Save other data
    generic_data = {}
    for k, v in place_details.items():
        if k != "photos" and k != "reviews":
            generic_data[k] = v
    with open(f"{data_path}/data.json", "w") as json_file:
        json.dump(generic_data, json_file, indent=4)

    # TODO: This is a hack. Use a JSON data loader.
    generic_data = TextLoader(f"{data_path}/data.json").load()
    generic_data_splits = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0,
        separators=[],
    ).split_documents(generic_data)

    website_url = place_details.get("website", "")
    website_data = WebBaseLoader(website_url).load()
    web_splits = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0,
        separators=[],
    ).split_documents(website_data)

    # Build vector database for restaurant
    embedding = OpenAIEmbeddings(openai_api_key=constants.OPENAI_API_KEY)
    for idx, documents in enumerate([reviews_splits, generic_data_splits, web_splits]):
        if idx == 0:
            index = Chroma.from_documents(
                documents=documents,
                embedding=embedding,
                persist_directory=data_path,
            )
        index.persist()
        restaurants[place_id] = {
            "index": index,
            "place_details": place_details,
        }


@app.get("/api/index")
def hello_world():
    return {"message": "Hello World"}


# TODO: move types to separate file
class Position(BaseModel):
    lat: float
    lng: float


@app.post("api/position")
async def receive_position(position: Position):
    lat = position.lat
    lng = position.lng

    # Write the coordinates to a JSON file
    with open('output/position.json', 'w') as f:
        json.dump({'lat': lat, 'lng': lng}, f)
    
    find_restaurant(position)

@app.post("api/restaurant")
def find_restaurant(position: Position):
    # Define the parameters
    lat = position.lat
    lng = position.lng
    radius = 100
    type = "restaurant"
    api_key = constants.GCP_API_KEY

    # Define the URL
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = f"?location={lat}%2C{lng}&radius={radius}&type={type}&key={api_key}"
    full_url = url + params

    # Make the API request
    response = requests.get(full_url)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])
        for place in results:
            print(f"Name: {place['name']}, Location: {place['geometry']['location']}")
        # Write the results to a JSON file
        with open('output/restaurant.json', 'w') as f:
            json.dump(results, f, indent=4)
        return results
    else:
        raise HTTPException(status_code=400, detail="Failed to find restaurant")
        
@app.post("api/user")
async def store_user_pref(request: Request):
    user_data = await request.json()
    print(user_data)
    with open('output/user.json', 'w') as f:
        json.dump(user_data, f)

# TODO: We can load the data from disk, we don't need to run this every time
save_restaurant_data(RESTAURANT)

if __name__ == "__main__":
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True)
    find_restaurant(Position(lat=51.5074, lng=0.1278))