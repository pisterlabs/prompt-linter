import json
from autogen.room_chat import RoomChat
from base_conversation import BaseConversation
from openai_harness import openai_call
import game_helpers

class NPCConversation(BaseConversation):
    def __init__(self, entity_id, source_spawnable_entity_id: int):
        super().__init__(entity_id, source_spawnable_entity_id)        

    def handle_prompt(self, promptInfo):
        self.prompt_prefix = f"You are {promptInfo.me.name}, {promptInfo.me.description}. "

        self.thinking = True
        prompt = f"You are in the room named {promptInfo.source_room.name}. Your name is {promptInfo.me.name}. Your bio is {promptInfo.me.description}.\n\n"

        # Add your NPC's character roleplaying logic here, for example:
        prompt += f"Room conversation:\n{game_helpers.get_chat_log(promptInfo.source_room.room_id)}.\n\nWhat would you say to continue this conversation? If you have nothing to say or this message was not directed at you give an empty response. Respond only with json in the format {{\"your_message\": \"Your response if any\"}}."

        message_json = None
        try:
            message_response = openai_call(prompt)            
            message_json = json.loads(message_response)
        except Exception as e:
            print("OpenAI Prompt Error: " + str(e))
            self.respond("Sorry, I can't respond right now.", promptInfo.is_tell)
            self.thinking = False
            return
        
        if message_json["your_message"] is not None:
            if message_json["your_message"] != "" and message_json["your_message"] != "Your response if any":                
                self.respond(message_json["your_message"].strip("\""), promptInfo.is_tell)
            self.thinking = False
        else:
            print("OpenAI Prompt Error: Unexpected response format")
            self.respond("Sorry, I can't respond right now.", promptInfo.is_tell)
            self.thinking = False
           
