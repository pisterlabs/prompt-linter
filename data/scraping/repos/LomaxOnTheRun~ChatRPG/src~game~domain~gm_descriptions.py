import openai

from . import utils
from .. import models


def get_one_player_game_intro(character: models.Character) -> str:
    """
    Get an intro description to an adventure for a given character.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Game Master running a fantasy RPG tabletop game for a "
                    f"single player. Their character is called {character.name} and "
                    f"they are a {character.race_name} {character.class_name}."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Describe the start of a new adventure. Do not describe anything "
                    "that will happen in the future. Only write a short paragraph."
                ),
            },
        ],
    )
    return response.choices[0]["message"]["content"]


def get_one_player_game_next_description(
    character: models.Character, game: models.OnePlayerGame
) -> str:
    """
    Get a description from the GM of a one-player game, responding the latest prompt
    from the player.
    """

    previous_messages = _get_previous_messages(game, max_num_turns=3)

    # Add a prompt to the final message to try and shape the response to be "better"
    final_prompt = (
        "Describe what happens next in the story. Only write a short paragraph."
    )
    previous_messages[-1]["content"] = previous_messages[-1]["content"] + final_prompt

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Game Master running a fantasy RPG tabletop game for a "
                    f"single player. Their character is called {character.name} and "
                    f"they are a {character.race_name} {character.class_name}."
                ),
            },
            *previous_messages,
        ],
    )
    return response.choices[0]["message"]["content"]


def _get_previous_messages(game: models.OnePlayerGame) -> list[dict[str, str]]:
    game_turns = models.OnePlayerGameTurn.objects.filter(game=game).order_by(
        "created_at"
    )

    last_turn = game_turns.last()

    # TODO: Return more previous turns
    return [
        {
            "role": "user",
            "content": f"{last_turn.description} Describe what happens next in the "
            "story. Only write a short paragraph.",
        }
    ]


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
    The GM is currently the AI (or "assistant"), while the player is the external
    "user", who is asking questions of the GM / "assistant".
    """

    return "assistant" if turn.character is None else "user"
