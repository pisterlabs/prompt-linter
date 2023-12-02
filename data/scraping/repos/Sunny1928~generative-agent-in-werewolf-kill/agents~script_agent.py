import requests
import threading
import logging
from openai import OpenAI , AzureOpenAI
import openai
import sys   
from pathlib import Path   
import time
import json
import datetime
from .agent import agent

class script_agent(agent):
    def __init__(self , api_json = None, game_info_path = None,
                 agent_name = "ScriptGame" , game_room = "ScriptGame"):

        # basic setting
        self.name = agent_name
        self.room = game_room
        self.logger : logging.Logger = logging.getLogger(f"{__name__}.{self.name}")
        self.logger_handler = []
        self.__logging_setting__()

        # openai api setting
        self.api_kwargs = {}
        try:
            self.__openai_init__(api_json)
        except:
            raise Exception("API Init failed")
        
        # game info 
        self.role = None
        self.player_id = None
        self.player_name = []
        self.game_data : list[dict]= None
        self.teamate = None

        # from file load a game
        self.__load_game_info__(game_info_path)
        # start the script game
        self.__start_script_game__()
    
    def __load_game_info__(self , game_info_path , const_player_name = True):
        with open(game_info_path , "r" , encoding="utf-8") as f:
            agent_info : dict[str:str] = json.loads(f.readline())
            # first line with agent info
            self.role = list(agent_info.values())[0]
            self.player_id = list(agent_info.keys())[0]
            # second line with player info 
            player_info :  dict[str: dict[str,str]] = json.loads(f.readline())
            self.teamate = []
            for id , info in player_info.items():
                if const_player_name:
                    self.player_name.append(f"Player{id}")
                else:
                    self.player_name.append(info["user_name"])
                # get the teammate info
                if id != self.player_id and info['user_role'] == "werewolf":
                    self.teamate.append(id)

            if self.role == "werewolf":
                pass


            self.game_data = [json.loads(info) for info in f.readlines() if "stage" in json.loads(info).keys()]
        
    def __start_script_game__(self):
        self.__start_game_init__(None)
        for data in self.game_data:
            self.logger.debug(data)
            self.__process_data__(data)

            for anno in data['announcement']: 
                if anno['operation'] == "game_over" : 
                    self.__game_over_process__(anno , 0)
                    break

    def __start_game_init__(self , room_data):
        self.logger.debug(f"game is started , this final room info : {room_data}")

    def __get_role__(self):
        pass    
    def __get_all_role__(self):
        pass
    def __check_game_state__(self, failure_cnt):
        pass
    def __game_over_process__(self, anno, wait_time):
        self.logger.info(f"Script game is over , {anno['description']}")

    def __send_operation__(self, data):
        self.logger.debug(f"Agent send operation : {data}")

    def __del__(self):
        pass