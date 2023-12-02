from fastapi import status, Body
import ipdb
import openai
from dotenv import load_dotenv
import os
import json

from fastapi.responses import JSONResponse
from master_ozz.ozz_query import Scenarios
from master_ozz.utils import ozz_master_root

from fastapi import APIRouter
router = APIRouter(
    prefix="/api/data",
    tags=["auth"]
)
# from fastapi import FastAPI
# router = FastAPI()

main_root = ozz_master_root()  # os.getcwd()
load_dotenv(os.path.join(main_root, ".env"))

# setting up FastAPI

# Loading the environment variables

@router.get("/test", status_code=status.HTTP_200_OK)
def load_ozz_voice():
    json_data = {'msg': 'test'}
    return JSONResponse(content=json_data)

@router.post("/voiceGPT", status_code=status.HTTP_200_OK)
def load_ozz_voice(api_key=Body(...), text=Body(...), self_image=Body(...)):
    # Test Queries with user and assistant and saving in conversation history as well as json file
    # text = [{"role": "system", "content": "You are a cute and smart assistant for kids."},
    #         {'role':'user','content': 'hey hootie tell me a story'}]
    # text = [  # future state
    #         {"role": "system", "content": "You are a cute and smart assistant for kids."},
    #         {'role':'user', 'content': 'hey hootie tell me a story'}, {'role':'assistant','content': 'what story would you like to hear'}, 
    #         {'role':'user','content': 'any kind of kid related'}
    #        ]
    ipdb.set_trace()

    def handle_response(text : str):
        ipdb.set_trace()
        # Kids or User question
        text_obj = text[-1]['user'] # user query 

        #Conversation History to chat back and forth
        conversation_history : list = [] if len(text) <= 1 else text

        first_ask = True if len(conversation_history) == 0 else False

        # Call the Scenario Function and get the response accordingly
        response = Scenarios(text_obj,conversation_history, first_ask=first_ask, conv_history=False)
        
        # For saving a chat history for current session in json file
        with open('fastapi/conversation_history.json','w') as conversation_history_file:
            json.dump(conversation_history,conversation_history_file)


        # update reponse to self   !!! well we are not using class methods so self doesn't work we just simply need to return response 
        # as functional based prototyping but if you have rest of the code then it will work according to the code
        text[-1].update({'resp': response})
        # text[-1] = response  # for normal response return without class

        return text

    text = handle_response(text)
    
    def handle_image(text, self_image):
        # based on LLM response handle image if needs to change
        self_image = '/Users/stefanstapinski/ENV/pollen/pollen/custom_voiceGPT/frontend/build/hootsAndHootie.png'

        return self_image

    self_image = handle_image(text, self_image)
    
    # audio_file = 'pollen/db/audio_files/file1.mp4'
    audio_file = '/Users/stefanstapinski/ENV/pollen/pollen/custom_voiceGPT/frontend/build/test_audio.mp3'
    
    json_data = {'text': text, 'audio_path': audio_file, 'page_direct': None, 'self_image': self_image}


    return JSONResponse(content=json_data)