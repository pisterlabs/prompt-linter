import random
from ..llm import openai, make_completion
from ..templates import get_template

print(make_completion([{ "role": "user", "content": "hello" }]))

def get_personality():
    get_template('./templates/personality.txt')


def get_random_position(width: int, height: int):
    return (random.randrange(0, width), random.randrange(0, height))

def make_new_player(width: int, height: int, players: list):
    position = None
    while position is None or position in [p.get('position') for p in players]:
        position = get_random_position(width, height)
    return {
            "position": position,
        }
    
def start_new_game(width: int, height: int):
    players = []
    for i in range(0, 5):
        players.append(make_new_player(width, height, players))

    return {
        "width": width,
        "height": height,
        "players": players,
    }
