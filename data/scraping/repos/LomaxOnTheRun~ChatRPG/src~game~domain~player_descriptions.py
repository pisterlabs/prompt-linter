import openai

from . import utils
from .. import models


def get_one_player_game_next_description(
    character: models.Character, game: models.OnePlayerGame
) -> str:
    """
    Get a description from the player of a one-player game, responding the latest
    prompt from the GM.
    """

    previous_messages = _get_previous_messages(game, max_num_turns=3)

    # Add a prompt to the final message to try and shape the response to be "better"
    final_prompt = "Describe how you react to this. Only write a short paragraph."
    previous_messages[-1]["content"] = previous_messages[-1]["content"] + final_prompt

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the only player of a fantasy RPG tabletop game, which is "
                    "being run by a Game Master. Your character is called "
                    f"{character.name} and they are a {character.race_name} "
                    f"{character.class_name}."
                ),
            },
            *previous_messages,
        ],
    )
    return response.choices[0]["message"]["content"]


def _get_previous_messages(
    game: models.OnePlayerGame, max_num_turns: int
) -> list[dict[str, str]]:
    """
    Return the previous messages in the "conversation" between the GM and the player.
    """

    lastest_game_turns = utils.get_latest_one_player_game_turns(game, max_num_turns)

    previous_messages = []
    for turn in lastest_game_turns:
        role = _get_role(turn)
        previous_messages.append({"role": role, "content": turn.description})

    return previous_messages


def _get_role(turn: models.OnePlayerGameTurn) -> str:
    """
    The player is currently the AI (or "assistant"), while the GM is the external
    "user", who is asking questions of the player / "assistant".
    """

    return "user" if turn.character is None else "assistant"
