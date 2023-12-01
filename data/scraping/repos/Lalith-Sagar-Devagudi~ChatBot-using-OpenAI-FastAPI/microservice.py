from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
import openai
import os
from pymongo import MongoClient
from pymongo.collection import Collection
from pydantic import BaseModel
import pyttsx3

tags_metadata = [
    {
        "name": "chat",
        "description": "You can chat here.",
    },
    {
        "name": "history",
        "description": "History of conversations by the user.",
    },
    {
        "name": "audio",
        "description": "Download audio file of the most recent OpenAI Bot response.",
    },
]

app = FastAPI( openapi_tags=tags_metadata, title="Chatbot Microservice API", description="API to chat with an AI chatbot", version="1.0.0")

origins = ["http://localhost:8001"]  # adjust this to match your actual domain

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the speech engine
engine = pyttsx3.init()

class ChatInput(BaseModel):
    """Data model for chat input."""
    user_id: str
    message: str
    audio: bool = False

# Set OpenAI API key
openai.api_key = "***********"  # set your OpenAI API key in environment variables

# Connect to MongoDB client
client = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
db = client['chatbot']  # Replace with your database name

@app.get("/", include_in_schema=False)
async def root():
    """
    Function to tell server is running whenevr the base URL http://127.0.0.1:8000 is opened.
    """
    return {"message": "Chatbot is running and you can access it at http://127.0.0.1:8000/docs."}

@app.post("/chat", tags=["chat"])
async def chat( user_id: str ="Enter your User ID ", message: str ="Chat here", audio: bool = False ) -> dict:
    """
    Function to handle chat with the bot.
    It saves conversation to the database, calls the AI model to get response,
    converts the response to speech and returns the AI response.
    \n **Audio is False = OFF by default, True = ON**
    """
    
    # Load conversation history from DB
    history = db.conversations.find_one({"user_id": user_id})
    if history is None:
        history = {"user_id": user_id, "conversation": []}
    
    # Use OpenAI API to generate a response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": message
        }],
        max_tokens=150
    )

    # Check if the response from OpenAI is empty or not
    if response is None or len(response.choices) == 0:
        raise HTTPException(status_code=500, detail="OpenAI response is empty")

    # Store the new messages in the conversation history
    new_messages = {
        "user": message,
        "assistant": response.choices[0].message['content'],
    }
    history['conversation'].append(new_messages)
    db.conversations.update_one({"user_id": user_id}, {"$set": history}, upsert=True)

    # Convert text to speech and speak the response only if audio was requested
    if audio:
            tts = gTTS(text=response.choices[0].message['content'], lang='en')
            tts.save("outputs/output.mp3")

            # Speak the response
            engine.say(response.choices[0].message['content'])
            engine.runAndWait()

    return {"message": response.choices[0].message['content']}  # return the AI response


@app.get("/history/{user_id}", tags=["history"])
async def get_history(user_id: str) -> FileResponse:
    """
    Function to get chat history for a given user id.
    It fetches the conversation from database, converts it to text file and returns as FileResponse.
    """
    history = db.conversations.find_one({"user_id": user_id})
    if history is None:
        raise HTTPException(status_code=404, detail="No conversation history found for this user")
    
    # Convert the conversation list into a single string, each message on a new line
    history_string = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in history["conversation"]])
    
    # Write the history string to a text file
    with open("outputs/history.txt", "w") as file:
        file.write(history_string)

    # Return the file as a response
    return FileResponse("outputs/history.txt", media_type="text/plain", filename=f"{user_id}_history.txt")

@app.get("/audio", tags=["audio"])
async def get_audio() -> FileResponse:
    """
    Function to return the most recent AI response as audio file.
    """
    return FileResponse("outputs/output.mp3", media_type="audio/mpeg", filename="output.mp3")
