import requests
import time
import os
import asyncio
import openai
import json


url_base = "https://cloud.leonardo.ai/api/rest/v1/"

card_template = """
{
    "name": "",
    "imageurl": "",
    "deck": "", // options are "Player" or "Boss"
    "description": "",
    "health": 0,
    "action": [
        {
        "amount": 1,
        "type": "",  // options are "Damage", "Heal", "HealthBuff", "HealBuff", "DamageBuff"
        "targetSize": "" // options are "Single", "All"
        }
    ],
    "effect": [
        {
        "name": "" // options are "Poison", "DivineShield", "Taunt", "Invisible"
        }
    ]
}
"""


class JsonGenerator:

    def __init__(self):
        self.OPEN_AI_KEY = os.environ.get('OPEN_AI_KEY')

    def get_json_as_dictionary(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[
                {"role": "system", "content": f"""
                    You are a game developer creating a new card game. You'll be returning only the JSON required to define
                    each card in the deck.  There will be 10 player cards and 1 boss enemy card.  You'll be given a theme and 
                    will create cards for the player and 1 boss card enemy that follow that theme.  The boss's relative power to the 
                    player cards should result in the boss losing 10% of it's life after fighting 5 player cards concurrently. 
                    Each card has a action and effect that should follow the theme. Please remember to ONLY return the JSON, no
                    other text or content, only JSON.  For the "effect" field, only fill that field for 50% of the cards.
                    For the "action" field, have 60% of the cards be "Damage" (with 70% of those being single target damage 
                    and 30% being multi-target damage) and 25% be "Heal" and 15% be a buff card.

                    card template: {card_template}
                    """},
                {"role": "user", "content": f"Theme: {prompt}"},
            ]
        )
        json_str = response['choices'][0]['message']['content']
        with open("game/game_file.json", "w") as json_file:
            json_file.write(json_str)
        return json.loads(json_str)
