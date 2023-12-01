import requests
import threading
import logging
import openai
import sys   
from pathlib import Path   
import time
import json
import numpy as np
import re
import time
import math
from sentence_transformers import SentenceTransformer, util
from ..agent import agent
from ..script_agent import script_agent
from .role import role , werewolf , seer , witch , hunter

class memory_stream_agent(agent):
    def __init__(self , api_json = "doc/secret/yui.key",  
                 server_url = "140.127.208.185" , agent_name = "Agent1" , room_name = "TESTROOM" , 
                 color = "f9a8d4" , prompt_dir = "doc/prompt/memory_stream"):
        
        super().__init__(api_json = api_json, server_url = server_url ,
                         agent_name = agent_name , room_name = room_name , 
                                       color = color) 
        
        # init long memory class & models
        self.long_memory : role = None
        # start the game
        self.day = None
        self.turn = 0

        self.prompt_dir = Path(prompt_dir)

    def get_info(self) -> dict[str,str]:
        
        return self.long_memory.get_long_memory_info()

    def __process_data__(self, data):
        """the data process."""
        operations = self.long_memory.update_stage(data)

        skip = False
        for operation in operations:
            self.__send_operation__(operation)
            if operation['operation'] == 'dialogue':
                skip = True
        
        if skip:
            self.__skip_stage__()


    def __start_game_init__(self , room_data):
        """the game started setting , update player name"""
        self.logger.debug(f"game is started , this final room info : {room_data}")
        self.player_name = [name for name in room_data["room_user"]]
        role_info = self.__get_role__()
        self.__get_all_role__()
        role_to_class = {
            "werewolf" : werewolf,
            "seer" : seer,
            "witch" : witch,
            "hunter" : hunter,
            "village" : role,
        }
        
        self.long_memory : role = role_to_class[self.role](self.prompt_dir , self.logger, self.api_kwargs)
        if self.role != "werewolf":
            self.long_memory.update_game_info(self.player_name , self.role)
        else:
            self.long_memory.update_game_info(self.player_name , self.role , role_info['game_info']['teamate'])

        self.__check_game_state__(0)


class memory_stream_agent_test(memory_stream_agent):
    def __init__(self , api_json = "doc/secret/yui.key",  
                 server_url = "140.127.208.185" , agent_name = "Agent1" , room_name = "TESTROOM" , 
                 color = "f9a8d4" , prompt_dir = "doc/prompt/memory_stream"):
        self.__reset_server__(server_url)
        
        super().__init__(api_json = api_json , server_url = server_url , 
                         agent_name = agent_name , room_name = room_name , 
                                       color = color , prompt_dir = prompt_dir) 
        
        # used for start game for test
        self.master_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX25hbWUiOiJ5dWkiLCJyb29tX25hbWUiOiJURVNUUk9PTSIsImxlYWRlciI6dHJ1ZSwiaWF0IjoxNjkwMzc5NTM0LCJleHAiOjE2OTkwMTk1MzR9.BEmD52DuK657YQezsqNgJAwbPfl54o8Pb--Dh7VQMMA"
        
        # set the game for test
        self.__setting_game()
        
        # start the game for test
        self.__start_server__()


    
    def __reset_server__(self , server_url):
        """for convenient test"""
        try :
            r = requests.get(f'{server_url}/api/reset' , timeout=5)
            # if r.status_code == 200:
            #     self.logger.debug("Reset Room Success")
            # else:
            #     self.logger.warning(f"Reset Room Error : {r.json()}")
        
        except Exception as e :
            self.logger.warning(f"__reset_server__ Server Error , {e}")
            
    def __start_server__(self):
        """for convenient test"""
        try :
            r = requests.get(f'{self.server_url}/api/start_game/{self.room}' , headers= {
                "Authorization" : f"Bearer {self.master_token}"
            })
            if r.status_code == 200:
                self.logger.debug("Start Game")
            else:
                self.logger.warning(f"Start Game : {r.json()}")
        
        except Exception as e :
            self.logger.warning(f"__start_server__ Server Error , {e}")
    
    def __setting_game(self):
        """for convenient test"""
        try :
            r = requests.post(f'{self.server_url}/api/room/{self.room}' , headers= {
                "Authorization" : f"Bearer {self.master_token}"
            }, json= {
                "player_num": 7,    
                "operation_time" : 10,
                "dialogue_time" : 10,
                "seer" : 1,
                "witch" : 1,
                "village" : 2,
                "werewolf" : 2,
                "hunter" : 1 
            })
            if r.status_code == 200:
                self.logger.debug("Setting Game Success")
            else:
                self.logger.warning(f"Setting Game Error : {r.json()}")
        
        except Exception as e :
            self.logger.warning(f"__setting_game Server Error , {e}")
    
    
class memory_stream_agent_script(script_agent):
    def __init__(self , api_json = None, game_info_path = None,
                agent_name = "ScriptGame" , game_room = "ScriptGame" , prompt_dir = "doc/prompt/memory_stream"):
        self.prompt_dir = Path(prompt_dir)
        super().__init__(api_json, game_info_path, agent_name , game_room)
    
    def __process_data__(self, data):
        """the data process."""
        operations = self.long_memory.update_stage(data)

        skip = False
        for operation in operations:
            self.__send_operation__(operation)
            if operation['operation'] == 'dialogue':
                skip = True
        
        if skip:
            self.__skip_stage__()


    def __start_game_init__(self , room_data):
        """the game started setting , update player name"""
        self.logger.debug(f"game is started , this final room info : {room_data}")
        role_to_class = {
            "werewolf" : werewolf,
            "seer" : seer,
            "witch" : witch,
            "hunter" : hunter,
            "village" : role,
        }
        
        self.long_memory : role = role_to_class[self.role](self.prompt_dir , self.logger, self.api_kwargs)
        if self.role != "werewolf":
            self.long_memory.update_game_info(self.player_name , self.role)
        else:
            self.long_memory.update_game_info(self.player_name , self.role , self.teamate)
