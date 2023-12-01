from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from elevenlabs import generate, set_api_key
import openai
import cohere
import os
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()

origins = ['http://localhost:3000']

chat_history = []
max_turns = 5

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# Construct the absolute path to the frontend/src/audios/ directory
AUDIOS_PATH = "../frontend/public/audios/"

# Define the AUDIO_PATH (this can remain as-is if it's a root-level path)
AUDIO_PATH = "./audios/"

@app.get("/voice/{query}")
async def voice_over(query: str):
    set_api_key(os.getenv("ELEVENLABS_API") )  # put your API key here

    count = 1

    audio_path = os.path.join(AUDIOS_PATH, f'{query[:4]}_{count}.mp3')
    file_path = f'{AUDIO_PATH}{query[:4]}_{count}.mp3'

    # Check if the file already exists
    
    while os.path.exists(audio_path):
        count += 1
        audio_path = os.path.join(AUDIOS_PATH, f'{query[:4]}_{count}.mp3')
        file_path = f'{AUDIO_PATH}{query[:4]}_{count}.mp3'
    


    audio = generate(
        text=query,
        voice='Bella',  # premade voice
        model="eleven_monolingual_v1"
    )

    print("Audio Path:", audio_path)
    print("File Path:", file_path)

    try:
        with open(audio_path, 'wb') as f:
            f.write(audio)

        return file_path

    except Exception as e:
        print(e)

        return {"error": "Failed to generate audio"}


@app.get("/chat/chatgpt/{query}")
def chat_chatgpt(query: str):
    openai.api_key = os.getenv("CHATGPT_API")  # put your API key here

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a conversation enthusiast, your goal is to provide a happy and enjoyable light conversation. Make sure your response is one sentence long."},
                {"role": "user", "content": query}
            ]
        )

        return response['choices'][0]['message']['content']

    except Exception as e:
        print(e)

        return {"error": "Failed to process chat query"}

@app.get("/chat/cohere/{query}")
def chat_cohere(query: str):
    cohere_key = os.getenv("COHERE_API")
    co = cohere.Client(cohere_key)  # put your API key here

    try:

        chat_history1 = [{"user_name": "User", "text": "What is my name?"},
                        {"user_name": "Chatbot", "text": "Your name is Gary"}]
        
        response = co.chat(
            message = query,
            model="command-light",
            chat_history=chat_history1,
        )

        return response.text


        # for _ in range(max_turns):

        #     response = co.chat(
        #         query,
        #         model="command-light",
        #         temperature=0.9,
        #         chat_history=chat_history
        #     )

        #     answer = response.text

        #     user_message = {"user_name": "User", "text": query}
        #     bot_message = {"user_name": "Chatbot", "text": answer}

        #     chat_history.append(user_message)
        #     chat_history.append(bot_message)
        
        return answer

    except Exception as e:
        print(e)

        return {"error": "Failed to process chat query"}