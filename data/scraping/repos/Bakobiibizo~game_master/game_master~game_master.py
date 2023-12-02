

from langchain.prompts import PromptTemplate
from game_master.datastax_vectordb import get_memory_module
from game_master.openai_llm import openai_llm_chain
from game_master.templates import get_selector
from game_master.broadcasts import game_broadcast
from helpers.choice_selector import get_selector
from helpers.uuid_generator import generate_uuids
from helpers.logger import get_logger




class GameMaster():
    def __init__(self):
        self.logger = get_logger()
        self.agent_modes = [
            "game_master"
            ]
        self.selector = get_selector
        self.user_id = 
        self.session_id = get_uuids(1)

    def set_user_id(user_id:str)->None:
        self.session_id = user_id
            
    def get_session_id(user_id:str)->None:
        

def start_game() -> tuple(str, str, str):
    logger.info("- Starting the game")
    game_modes = selector(agent_modes)
    template, choice = selector(game_modes)
    game_broadcast(choice)
    return template, choice
    
def run_llm(session, template, ):
    memory = get_memory_module(session)
    logger.info("- Constructing prompt")
    prompt = PromptTemplate(template=template, input_variables=["chat_history", "human_input"])
    
    openai_llm_chain(memory, prompt)

    return memory

def main():
    template, choice = start_game()
    game_broadcast(choice)
    session_id = get_session_id(USER_ID)
    run_llm(session, template)
    
    