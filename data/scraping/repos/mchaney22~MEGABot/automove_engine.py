import openai
import os
from dotenv import load_dotenv
import json
import pyanswers
import BadCompany

class AutomoveEngine():
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv('openai_token')
        self.cw = BadCompany.CW()
    
    def comment(self, msg):
        pass

    def query_automove(self, question):
        pass
    
    # given the player and the question, return the name of the applicable move
    def id_move(self, ctx, question):
        # get the player's name from ctx
        player = ctx.message.author.name
        # get the player's moves
        moves = self.cw.player_to_moves(player)
        


