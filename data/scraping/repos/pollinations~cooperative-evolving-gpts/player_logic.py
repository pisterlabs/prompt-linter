from typing import List, Optional
from models import Player, Message, ChatGPTMessage
import openai
import random
from sampledata import GENES, NAMES, INITIAL_PROMPT  # Assuming these are defined in your sampledata.py
import sys
import json
from utils import mermaid_print


# Define available functions that ChatGPT can call
functions = [
    {
        "name": "submit_guess",
        "desniption": "When you think you know enough secrets, submit your guess. You will get notified how many secrets you got right, so you can use this to test if someone is telling the truth.",
        "parameters": {
            "type": "object",
            "properties": {
                "secrets": {
                    "type": "string",
                    "description": "Comma separated list of secrets you think are true.",
                },
            },
            "required": ["secrets"],
        },
    },
    {
        "name": "send_message",
        "description": "Send a message to another player.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "The name of the player you want to send a message to.",
                },
                "message": {
                    "type": "string",
                    "description": "The message you want to send.",
                },
            },
            "required": ["to", "message"],
        },
    }
]

# Create a player
# name: name of the player
# dna: list of genes
# required_secrets: number of secrets each player needs to find to win
# secret: secret of the player
# names: names of all players
def create_player(name:str, dna:List[str], required_secrets:int, secret:str, names:List[str]) -> Player:
    mermaid_print(f"participant {name}")
    mermaid_print(f"Note over {name}: {','.join(dna)}")
    mermaid_print(f"Note right of {name}: Initial secret: {secret}")
    # print dna as a note
    other_names = list(set(names) - {name})

    initial_message: ChatGPTMessage = {
        "role": "system",
        "content": INITIAL_PROMPT.format(name=name, dna="\n".join(dna), secret=secret, required_secrets=required_secrets-1, names=", ".join(other_names))
    }
    
    return {
        "name": name,
        "dna": dna,
        "secret": secret,
        "points": 100,
        "history": [
            initial_message
        ]
    }



# Make a move for a player (doesn't mutate the player but returns a new one)
def make_move(player: Player) -> Player:
    completion = openai.ChatCompletion.create(
        model= "gpt-3.5-turbo-0613",#"gpt-4-0613",#"
        messages=player["history"],
        functions=functions,
        temperature=0.3,
        frequency_penalty=0.9,
    )

    response = completion.choices[0].message.to_dict()
    player_with_new_history = add_to_history(player, response)
    if "function_call" not in response:
        print("Response did not include function call. Trying again.", file=sys.stderr)
        error_message: ChatGPTMessage = {"role": "user", "content": "Error. Your response did not contain a function call."}
        player_with_new_history = add_to_history(player, error_message)
        return make_move(player_with_new_history)
    return player_with_new_history



# Add an observation to a player's history
# player: player
# observation: observation
# returns: player
def add_to_history(player:Player, observation:ChatGPTMessage):
    """Add an observation to a player's history."""
    new_player = {
        **player,
        "history": player["history"] + [observation]
    }
    return new_player




# Get the outbox of a player. Secreits is used to check if the message contains any of the secrets for logging.
# player: player
# secrets: list of secrets
# returns: outbox
def get_outbox(player: Player, secrets: List[str]) -> Optional[Message]:
    move = get_last_move(player)
    if move is not None and move["name"] == "send_message":
        try:
            message = json.loads(move["arguments"])
            message["from"] = player["name"]
            # if the message' text contains any of the secrets, print sender, receiver and secret
            mermaid_print(f"{message['from']} -> {message['to']}: {message['message']}")
            for secret in secrets:
                if secret in message["message"]:
                    mermaid_print(f"Note over {message['from']},{message['to']}: SECRET: {secret}")
            return message
        except:
            # print to stderr
            print("Error parsing message arguments", move["arguments"], file=sys.stderr)
    return None


# Get the last move of a player
# player: player
# returns: last move
def get_last_move(player:Player):
    if len(player["history"]) == 0:
        return None
    if "function_call" not in player["history"][-1]:
        return None
    return player["history"][-1]["function_call"]

# Add points to a player
# player: player
# points: number of points to add
# returns: player
def add_points(player:Player, points:int):
    return {
        **player,
        "points": player["points"] + points
    }


