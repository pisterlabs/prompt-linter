import streamlit as st
from utils import load_from_jsonl, save_to_jsonl
from game import start_game
import openai

DATA_PATH = "data.jsonl"
MODEL = "gpt-3.5-turbo"

st.text("AttentionX Mystery Game")

input_group = st.text_input("Group name")

data = load_from_jsonl(DATA_PATH)
groups = []
game_settings = None
chat_history = [
    {"role": "system", "content": "Play a game with the players."},
]

if data and "group" in data[0]:
    groups.extend([d["group"] for d in data])

if len(input_group) > 0:
    if input_group not in groups:
        players = st.text_input("Add players in this format: player1, player2, player3")
        if players and len(players) > 0:
            game_settings = start_game(characters=players)
            save_to_jsonl(
                [
                    {
                        "group": input_group,
                        "players": players.split(", "),
                        "settings": game_settings,
                    }
                ],
                DATA_PATH,
            )
            chat_history.extend(
                [
                    {
                        "role": "user",
                        "content": """
                    Let's play a detective game with some players!
                    I will tell you which player's turn it is by their names.
                    You will be an exclusive sixth player.

                    The game will consist of five characters, five places, and five tools.
                    One specific character has commited crime at one specific place with one specific tool. The goal of players is to correctly guess the tuple of correct character, place, and tool within 15 player turns in total.
                    If a player guesses the correct match he/she wins. If no player matches the answer within 15 turns in total, you win.

                    On every turn, a player will either ask you one question, or try to guess the answer. You will either answer the question (without revealing the answer) or respond if the guess is right or wrong.
                    """,
                    },
                    {"role": "assistant", "content": "Okay! Let's go!"},
                ]
            )
    else:
        st.error("Use another group name!")

if game_settings is not None:
    st.text(f"Characters: {game_settings['characters']}")
    st.text(f"Places: {game_settings['places']}")
    st.text(f"Tools: {game_settings['tools']}")
    st.text(f"Story: {game_settings['story']}")

    turn = 1
    player_index = 0
    players = game_settings["characters"].split(",")

    while turn <= 15:
        def on_submit_chat():
            if player_index + 1 < len(players):
                player_index += 1
            else:
                player_index = 0
            if turn == 15:
                st.text("Game over! AI wins!")
                return
            else:
                turn += 1
            chat_history.append({"role": "user", "content": chat_input})
            chat_response = (
                openai.ChatCompletion.create(
                    model=MODEL,
                    messages=chat_history,
                )
                .choices[0]
                .message.content
            )

            chat_history.append({"role": "assistant", "content": chat_response})

        st.text(f"Turn #{turn}: {players[player_index]}'s turn!")
        chat_input = st.chat_input("Ask something", on_submit=on_submit_chat)
