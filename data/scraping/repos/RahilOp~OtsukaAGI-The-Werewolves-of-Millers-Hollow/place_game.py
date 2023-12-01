import pygame
from utils1 import create_new_memory_retriever, LLM
from langchain.experimental.generative_agents import GenerativeAgentMemory
from game_gen_memory import GameGenerativeAgentMemory

class Place():
    def __init__(self, name, description, file_path, x, y, x_bottom, y_bottom,name_japan,desc_japan):

        self.x = x
        self.y = y
        self.x_bottom = x_bottom
        self.y_bottom = y_bottom
        self.height = y_bottom - y
        self.width = x_bottom - x
        self.rect = pygame.Rect(x, y, self.width, self.height)
        self.name = name
        self.description = description
        self.file_path = file_path
        self.name_japan = name_japan
        self.desc_japan = desc_japan
        
        
        self.objects = {}
        
        self.sabotage_memory = ""
        
        self.history = GameGenerativeAgentMemory(
        llm=LLM,
        file_path=self.file_path,
        memory_retriever=create_new_memory_retriever(),
        verbose=False,
        reflection_threshold=8 # we will give this a relatively low number to show how reflection works
    )
    
    def add_history(self, input:str):
        self.history.add_memory(input)

