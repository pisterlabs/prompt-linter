
import math
from typing import Optional, Type

from langchain import LLMMathChain
from langchain.callbacks.manager import (AsyncCallbackManagerForToolRun,
                                         CallbackManagerForToolRun)
from langchain.chat_models import ChatOpenAI
from langchain.experimental.plan_and_execute import (PlanAndExecute,
                                                     load_agent_executor,
                                                     load_chat_planner)
from langchain.tools import BaseTool

from ..entity import Entity

from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import json

from .. import klogger as kl
from . import pathfinder
from ..settings import *


KAFKA_ADDRESS = "localhost:9092"

class Agent(Entity.Entity):
    
    delayed_actions = []
    
    def __init__(self, agent_config, groups, level, map, agents):
        super().__init__(agent_config, groups, level, agents)
        self.needs = {
            "hunger": 1.0,
            "thirst": 1.0,
            "sleep": 1.0,
            "toilet": 1.0,
            "social": 1.0
        }
        # self.tool = StructuredTool().from_function()
        self.map = map
        self.place_memory = {}
        
        self.time_from_social = 0
        self.time_from_eat = 0
        self.need_modif = 0.1
        self.kafka_producer = KafkaProducer(bootstrap_servers=KAFKA_ADDRESS)
        self.kafka_consumer = KafkaConsumer(agent_config["name"], bootstrap_servers=KAFKA_ADDRESS, consumer_timeout_ms=5, value_deserializer=lambda m: json.loads(m.decode('ascii')))
        self.health = 100
        kl.klog(self.name, "init", "Agent loaded")




        
        self.pathfinder = pathfinder.Pathfinder(self.name, self.map) 


    def get_path(self, start, end):
        kl.klog(self.name, "get_path", f"start: {start}, end: {end}", level="DEBUG")
        path = self.pathfinder.astar(start, end)
        kl.klog(self.name, "get_path", f"path: {path}", level="DEBUG")
        return path        




    def interact(self, msg):
        pass

    def save_place(self, entity):
        if not self.place_memory.keys().__contains__(entity.name):
            self.place_memory[entity.name] = entity.real_position

    def get_known_place(self):
        """Get all known places in memory from the town. Can be used to retrive all the possible places to go to.

        Returns:
            str: all known places in memory
        """
        # print(f"self.place_memory: {self.place_memory}")
        
        
        return json.dumps(self.place_memory)

    def move_to_known_place_by_name(self, name: str, action_id: str):
        """Move to known place by name

        Returns:
            str: succes or error
        """
        kl.klog(self.name, "move_to_known_place_by_name", f"name: {name}, action_id: {action_id}, memory:  {self.place_memory}")
        if self.place_memory.keys().__contains__(name):
            entity_pos = self.place_memory[name]
            calculated_path = self.pathfinder.astar(self.real_position, entity_pos)
            self.delayed_actions.append({"action": "move", "path": calculated_path, "action_id": action_id, "place_name": name})
            return "delayed"
        return "there is no known place with this name"


    

    
    def get_health(self):
        """ Get health status

        Returns:
            str: health status
        """
        if self.health < 25:
            return "I'm dying"
        if self.health < 50:
            return "I'm very sick"
        if self.health < 75:
            return "I'm sick"
        if self.health < 100:
            return "I'm ok"

    def calculate_needs(self):

        self.needs["hunger"] -= self.need_modif
        self.needs["thirst"] -= self.need_modif
        # TODO add day_night cycle
        self.needs["sleep"] -= self.need_modif
        self.needs["toilet"] -= self.need_modif * \
            (24 / (self.time_from_eat + 1))
        self.needs["social"] -= self.need_modif * \
            (24 / (self.time_from_social + 1))

    def get_surroundings(self):
        surrondings = {}

        distance = 1500
        # print(f"self.leve {self.level}")
        for enitity in self.level:
            # print(f"enitity: {enitity}")
            if math.hypot(enitity[0] - self.position[0], enitity[1] - self.position[1]) < distance and self.level[enitity].name != "Grass":
                 
                 surrondings[self.level[enitity].name] = {
                     "name" : self.level[enitity].name,
                     "position" : self.level[enitity].position}
                 self.save_place(self.level[enitity])
        kl.klog(self.name, "get_surroundings", f"surrondings: {surrondings}")
        return surrondings

    def calculate_health(self):
        hp_modif = 0.0001
        self.health += (self.needs["hunger"] * hp_modif) + (self.needs["thirst"] * hp_modif) + (
            self.needs["sleep"] * hp_modif) + (self.needs["toilet"] * hp_modif) + (self.needs["social"] * hp_modif)

    def think(self):
        pass

    def create_initial_memory(self):
        return f"You playing a text based rpg in a medival age. You are a {self.name} and you are in a town. You have a {self.inventory} in your inventory. Your only goal is to stay healthy. This is your backstory: {self.description}"


    def get_action(self):
        #print(f"get_action: {self.name}")
        for msg in self.kafka_consumer:
            action_id = f"{self.name}{msg.value['timestamp']}"
            kl.klog(self.name, "get_action", f"msg: [{msg.value['action']}]")
            action = msg.value["action"]
            
            
            if action == "get_known_places":
                response = self.get_known_place()
                
            elif action == "move_to_known_place_by_name":
                response = self.move_to_known_place_by_name(msg.value["arg"], action_id)
            elif action == "get_health":
                response = self.get_health()
            elif action == "get_memory":
                response = self.create_initial_memory()
            elif action ==  "get_surroundings":
                response = self.get_surroundings()
            elif action == "interact":
                response = self.interact(msg.value["arg"])
            else :
                response = f"Action: {msg.value['action']} is unimplemented"
            
            
            kl.klog(self.name, "get_action", f"response: {response}")
            #Return if the action is delayed
            if response == "delayed":
                return
                        
            data = json.dumps(response)
            future = self.kafka_producer.send(action_id, data.encode('utf8'))
            try:
                record_metadata = future.get(timeout=10)
            except KafkaError:
                kl.klog(self.name, "get_action", f"KafkaError: {KafkaError}", level="ERROR")
                pass
           
    def move_cycle(self):
        if self.delayed_actions.__len__() > 0:
            kl.klog(self.name, "move_cycle", f"self.delayed_actions: {self.delayed_actions}", level="DEBUG")
            action = self.delayed_actions[0]
            if action["action"] == "move":
                kl.klog(self.name, "move_cycle", f"self.delayed_actions: {self.delayed_actions}", level="DEBUG")
                if self.real_position == action["path"][0]:
                    
                    self.delayed_actions.pop(0)
                    kl.klog(self.name, "move_cycle", f"self.delayed_actions: {self.delayed_actions}", level="DEBUG")
                else:
                    self.real_position = action["path"][0]
                    self.position = (
                        self.real_position[0] * TILESIZE,
                        self.real_position[1] * TILESIZE
                    )
                    self.rect = self.image.get_rect(topleft=self.position)
                    self.delayed_actions[0]["path"].pop(0)
                    kl.klog(self.name, "move_cycle", f"self.delayed_actions: {self.delayed_actions}", level="DEBUG")
                        

    def update(self):
        #print(f"update: {self.name}")
        self.calculate_needs()
        self.calculate_health()
        self.get_action()
        self.move_cycle()