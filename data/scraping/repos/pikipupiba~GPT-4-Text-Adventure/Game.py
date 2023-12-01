#TODO:
# 1. track tokens with Game objects (__dict__ anyone?)
# 2. auto switch to gpt-3.5-turbo-16k-0613 when context is too long
# 3. add a 3rd item to messages indicating real_model
# 4. Output combat and stats in real time. (use a schema for this yeehaw)
# 5. Add a little text spinner to the chatbot while it's thinking
#    - Can use emojis! :D
import json,re,os,time
from datetime import datetime

import gradio as gr
import openai
from loguru import logger
from PythonClasses.Helpers.helpers import randomish_words
from PythonClasses.Helpers.helpers import generate_dice_string

from PythonClasses.Game.Turn import Turn
from PythonClasses.Game.SystemMessage import SystemMessage
from PythonClasses.Game.UserMessage import UserMessage

# from PythonClasses.Game.Speech import LLMStreamProcessor

from PythonClasses.LLM.LLM import LLM
from PythonClasses.LLM.LLMModel import LLMModel, total_tokens
from PythonClasses.Game.Speech import LLMChunker
from PythonClasses.Game.CompleteJson import CompleteJson


class Game:
# The `Game` class  represents a game session. It keeps track of the game
# state, history of turns, and provides methods for interacting with the game. It also includes
# methods for rendering the game story, undoing turns, retrying turns, clearing the history,
# restarting the game, and submitting user messages. Additionally, it includes a method for
# streaming predictions from the language model.

    START, STOP, PREDICTING, AWAITING_USER = range(4)

    GAMES = {}

    # Initialize a new Game object for each active game.
    def __init__(self, game_name: str, history: [], system_message: str):
        logger.debug(f"Initializing Game: {game_name}")
        self.state = Game.START
        self.game_name = game_name
        self.dev = False
        self.audio = LLMChunker(game_name)
        self.audio_file = None

        self.llm_model = LLMModel()
        
        self.history = []

        intro_json = {
            "type": "normal",
            "model": "gpt-4-0613",
            "system_message": SystemMessage.inject_schemas(system_message, 1),
            "display": history[0],
            "raw": history[0],
            "stats": {
                "DAY": "Monday",
                "ITEM": [],
                "RELATIONSHIP": [],
            },
            "combat": [],
            "execution": {},
        }

        self.history.append(Turn(intro_json))

        choose_items_string = game_name + "\n" + "{Greet me and ask me to choose items}"

        choose_items_json = {
            "type": "normal",
            "model": "gpt-4-0613",
            "system_message": SystemMessage.inject_schemas(system_message, 2),
            "display": [game_name, None],
            "raw": [choose_items_string, None],
            "stats": {
                "DAY": "Monday",
                "ITEM": [],
                "RELATIONSHIP": [],
            },
            "combat": [],
            "execution": {},
        }

        self.history.append(Turn(choose_items_json))
        
        Game.GAMES[game_name] = self

    def compile_game_stats():
        from PythonClasses.Game.FileManager import FileManager

        final_stats = {}

        for game_name, game in Game.GAMES.items():
            final_stats[game_name] = Game._stats(game_name)

        directory_path = os.path.join(FileManager.DATA_FOLDER, "final_stats")
        os.makedirs(directory_path, exist_ok=True)

        with open (os.path.join(directory_path, "final_game_stats.json"), "w") as f:
            json.dump(final_stats, f, indent=4)
        
        return final_stats
    
    def reset_games():
        for game_name, game in Game.GAMES.items():
            del Game.GAMES[game_name]

    def start(game_name: str):
        logger.info(f"Starting Game: {game_name}")
        hide = gr.update(visible=False)
        show = gr.update(visible=True)
        # current_game = Game.GAMES[game_name]
        return Game.render_story(game_name) + [game_name, hide, hide, show, show, show]

    def __del__(self):
        logger.debug("Deleting Game")
        del self
        # delete any empty games
        for game_name, game in Game.GAMES.items():
            if game is None:
                del Game.GAMES[game_name]

    # Access the game object by name
    def _(game_name: str):
        return Game.GAMES[game_name]
    # # Access audio object
    # def _audio(game_name: str):
    #     return Game._(game_name).audio
    # def get_next_audio(game_name: str):
    #     return Game._audio(game_name).get_next_audio()
    
    # Access the game history by name
    def _history(game_name: str):
        return Game.GAMES[game_name].history
    def _history_to_dict(game_name: str):
        return [turn.__dict__() for turn in Game.GAMES[game_name].history]
    def _dict_to_history(game_name: str, history_dict_array: []):
        Game.GAMES[game_name].history = [Turn(turn) for turn in history_dict_array]
    def _num_turns(game_name: str):
        return len(Game.GAMES[game_name].history)
    
    # Chatbot history to display to the player
    def _display_history(game_name: str):
        return [turn.display for turn in Game.GAMES[game_name].history]
    # OpenAI history for the language model
    def _raw_history(game_name: str):
        return [turn.raw for turn in Game.GAMES[game_name].history]

    # Access the last turn by name
    def _last_turn(game_name: str):
        return Game.GAMES[game_name].history[-1]
    def _prev_turn(game_name: str):
        return Game.GAMES[game_name].history[-2]
    def _last_display(game_name: str):
        return Game.GAMES[game_name].history[-1].display
    def _last_raw(game_name: str):
        return Game.GAMES[game_name].history[-1].raw
    def _stats(game_name: str):
        return Game.GAMES[game_name].history[-1].stats
    def _combat(game_name: str):
        return Game.GAMES[game_name].history[-1].combat
    
    # Update the interface
    def render_story(game_name: str):
        display_history = Game._display_history(game_name)
        stats = Game._stats(game_name)

        # for item in stats.get("ITEM", []):
        #     if '(' in item:
        #         before_parenthesis, parenthesis, after_parenthesis = item.partition('(')
        #         return f'<p style="color:green">{before_parenthesis}</p>{parenthesis}{after_parenthesis}'
        #     else:
        #         return item

        day_box = stats.get("DAY")
        item_box = '\n'.join(stats.get("ITEM"))
        relationship_box = '\n\n'.join(stats.get("RELATIONSHIP"))

        # Sent to config tab for debugging
        turn_dict = Game._last_turn(game_name).__dict__()

        # Speech
        # speech = Game._audio(game_name).get_next_audio()

        if Game._(game_name).state == Game.AWAITING_USER:
            submit_button = gr.update(interactive=True, value="Do it!")
            user_message = gr.update(interactive=True, placeholder="What will you do?")
        else:
            submit_button = gr.update(interactive=False, value="Hold on...")
            user_message = gr.update(interactive=False, placeholder="Predicting...")

        if Game._(game_name).dev == True:
            gm_tab = gr.update(visible=True)
            config_tab = gr.update(visible=True)
        else:
            gm_tab = gr.update(visible=False)
            config_tab = gr.update(visible=False)

        # execution_json = Game._(game_name).llm_model.tokens
        num_games = len(Game.GAMES)

        for game in Game.GAMES.values():
            if game.dev:
                num_games -= 1

        if num_games == 0:
            num_games = 1

        gpt_versions = ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613", "gpt-4-0613", "gpt-4-32k-0613"]
        categories = ["total", "prompt", "completion"]

        game_average = {}

        for version in gpt_versions:
            game_average[version] = {}
            for category in categories:
                game_average[version][category] = {
                    "count": total_tokens[version][category]["count"] / num_games,
                    "cost": total_tokens[version][category]["cost"] / num_games,
                    "tpm": sum([game.llm_model.tokens[version][category]["tpm"] for game_name, game in Game.GAMES.items()]) / num_games,
                    "cpm": sum([game.llm_model.tokens[version][category]["cpm"] for game_name, game in Game.GAMES.items()]) / num_games
                }
        

        return [
            display_history,
            day_box,
            item_box,
            relationship_box,
            submit_button,
            user_message,
            turn_dict,
            gm_tab,
            config_tab,
            total_tokens["gpt-4-0613"],
            total_tokens["gpt-4-32k-0613"],
            total_tokens["gpt-3.5-turbo-0613"],
            total_tokens["gpt-3.5-turbo-16k-0613"],
            game_average["gpt-4-0613"],
            game_average["gpt-4-32k-0613"],
            game_average["gpt-3.5-turbo-0613"],
            game_average["gpt-3.5-turbo-16k-0613"],
            Game.GAMES[game_name].llm_model.last_turn_tokens["gpt-4-0613"],
            Game.GAMES[game_name].llm_model.last_turn_tokens["gpt-4-32k-0613"],
            Game.GAMES[game_name].llm_model.last_turn_tokens["gpt-3.5-turbo-0613"],
            Game.GAMES[game_name].llm_model.last_turn_tokens["gpt-3.5-turbo-16k-0613"],
            Game._(game_name).audio_file,
        ]
    
    def undo(game_name: str):
        logger.info(f"Undoing last turn: {game_name}")
        if len(Game._history(game_name)) > 0:
            del Game._history(game_name)[-1]
        return Game.render_story(game_name)
    def retry(game_name: str):
        logger.info("Retrying turn")
        Game._last_display(game_name)[1] = None
        Game._last_raw(game_name)[1] = None
        Game._last_turn(game_name).stats = Game._prev_turn(game_name).stats
        Game._last_turn(game_name).combat = []
        Game._last_turn(game_name).execution = {}
        return Game.render_story(game_name)
    def clear(game_name: str):
        logger.info("Clearing history")
        for turn in Game._history(game_name):
            del turn
        return Game.render_story(game_name)
    def restart(game_name: str):
        logger.info("Restarting game")
        Game.clear(game_name)
        return Game.start(game_name)
    
    def submit(game_name: str, message: str = "", system_message: str = None, model: str = None, system_select: str = None, schema_select: str = None):
        """
        This function is called when the user submits a message.
        """
        logger.debug(f"Submitting message: {message}")

        # TODO: USER MESSAGE CLASS TO FORMAT IT WITH DICE ROLLS AND SUCH??
        # # Add dice roll to the end of the user message
        # if "intRollArray" not in self.raw_history[-1][0]:
        #     self.raw_history[-1][0] += f'\n\n{generate_dice_string(10)}'

        while(Game._(game_name).state == Game.PREDICTING):
            time.sleep(1)

        if Game._num_turns(game_name) > 2:
            dice_string = generate_dice_string(5)
        else:
            dice_string = ""

        

        complete_user_message = f'{message}\n{dice_string}'

        complete_user_message += "\n{remember to update items, relationships, and day}"
        complete_system_message = SystemMessage.inject_schemas(system_message, Game._num_turns(game_name))

        new_turn_json = {
            "type": "normal",
            "model": model,
            "system_message": complete_system_message,
            "display": [message, None],
            "raw": [complete_user_message, None],
            "stats": Game._stats(game_name).copy(),
            "combat": [],
            "execution": {},
        }

        if "Try again! Encountered an exception:" in Game._last_display(game_name)[0]:
            Game.undo(game_name)

        if (Game._num_turns(game_name) == 0) or ((Game._last_raw(game_name)[1] is not None) and (len(Game._last_raw(game_name)[1]) > 0)):
            Game._history(game_name).append(Turn(new_turn_json))

        return [""] + Game.render_story(game_name)

    
    def stream_prediction(game_name: str, audio_speed:int = 125):

        try:
            new_day = True
            count_tokens = True

            if "dddeeevvv" in Game._last_display(game_name)[0]:
                del Game._history(game_name)[-1]
                Game._(game_name).dev = not Game._(game_name).dev
                new_day = False
                count_tokens = False

            Game._(game_name).state = Game.PREDICTING
            new_day_counter = 0

            while(new_day and new_day_counter < 3):
                new_day_counter += 1
                new_day = False
                    
                logger.info("Streaming prediction")
                current_turn = Game._last_turn(game_name)

                model = current_turn.model
                system_message = current_turn.system_message
                raw_history = Game._raw_history(game_name)

                if len(Game._history(game_name)) > 2:
                    Game._history(game_name)[-2].system_message = ""

                Game._last_raw(game_name)[1] = ""
                Game._last_display(game_name)[1] = ""

                schema_delimiter = r'\.\.\s*[A-Z]+\s*\.\.'  # regex pattern to find schema delimiters
                schema_name = None
                item_index = None
                temp_string = ""

                for chunk in LLM.predict(model, system_message, raw_history, Game._(game_name).llm_model):
                    if len(chunk["choices"][0]["delta"]) == 0:
                        break

                    content = chunk["choices"][0].get("delta", {}).get("content")

                    if content is None:                 
                        continue

                    Game._last_raw(game_name)[1] += content

                    # If not streaming, look for opening tag in the unprocessed content
                    if not schema_name:
                        Game._last_display(game_name)[1] += content
                        # Game._audio(game_name).process_data(content)

                        opening_match = re.search(schema_delimiter, Game._last_display(game_name)[1])
                        if opening_match:
                            schema_name = opening_match.group(0).strip(" .")
                            Game._last_display(game_name)[1] = Game._last_display(game_name)[1][:opening_match.start()]
                            content = content.rstrip('. \n')

                            if schema_name == "DAY":
                                Game._last_turn(game_name).stats["DAY"] = ""
                            elif schema_name == "ITEM":
                                if not "ITEM" in Game._last_turn(game_name).stats:
                                    Game._last_turn(game_name).stats["ITEM"] = []
                            elif schema_name == "RELATIONSHIP":
                                if not "RELATIONSHIP" in Game._last_turn(game_name).stats:
                                    Game._last_turn(game_name).stats["RELATIONSHIP"] = []
                            elif schema_name == "HIDE":
                                pass
                            else:
                                logger.error(f"Unknown schema: {schema_name}")
                                schema_name = None


                    if schema_name == "DAY":
                        Game._last_turn(game_name).stats["DAY"] += content
                        closing_match = re.search(schema_delimiter, Game._last_turn(game_name).stats["DAY"])
                        if closing_match:
                            schema_name = None
                            Game._last_turn(game_name).stats["DAY"] = Game._last_turn(game_name).stats["DAY"][:closing_match.start()]

                            # Extract the number from the day string
                            numbers_in_day = re.findall(r'\d+', Game._last_turn(game_name).stats["DAY"])
                            Game._last_turn(game_name).time_left = int(numbers_in_day[-1]) if len(numbers_in_day) > 0 else 0
                            if Game._last_turn(game_name).time_left <= 0:
                                logger.info("Day over")
                                new_day = True
                            
                    
                    elif schema_name == "HIDE":
                        temp_string += content
                        closing_match = re.search(schema_delimiter, temp_string)
                        if closing_match:
                            schema_name = None
                            temp_string = ""

                    elif (schema_name is not None) and (item_index is None):
                        temp_string += content
                        # See if exactly 1 item in items_array matches the content.
                        # Check if the start of any item in the array matches the content
                        # matching_indices = [index for index, item in enumerate(Game._last_turn(game_name).stats[schema_name]) if item.lower().startswith(temp_string.lower())]
                        compare_length = min(4, len(temp_string))
                        matching_indices = [index for index, item in enumerate(Game._last_turn(game_name).stats[schema_name]) 
                        if item[:compare_length].lower() == temp_string[:compare_length].lower()]
                        if len(matching_indices) == 0:
                            # # If no match, append the content to the end of the array
                            # item_index = len(Game._last_turn(game_name).stats[schema_name])
                            # Game._last_turn(game_name).stats[schema_name].append(temp_string)

                            # If no match, insert the content at the beginning of the array
                            item_index = 0
                            Game._last_turn(game_name).stats[schema_name].insert(item_index, temp_string)
                        elif len(matching_indices) == 1 and len(temp_string) > 4:
                            # If a match is found, replace the item at the first matching index
                            item_index = matching_indices[0]
                            Game._last_turn(game_name).stats[schema_name][item_index] = temp_string

                    elif (schema_name is not None) and (item_index is not None):
                        # If we already found the item index, just append the content to the item
                        Game._last_turn(game_name).stats[schema_name][item_index] += content
                        closing_match = re.search(schema_delimiter, Game._last_turn(game_name).stats[schema_name][item_index])
                        if closing_match:
                            Game._last_turn(game_name).stats[schema_name][item_index] = Game._last_turn(game_name).stats[schema_name][item_index][:closing_match.start()].strip(". \n")

                            if schema_name == "RELATIONSHIP":
                                first_line  = Game._last_turn(game_name).stats[schema_name][item_index].split('\n', 1)[0]
                                numbers = re.findall(r'\d+', first_line)
                                if numbers:
                                    last_number = int(numbers[-1])
                                    if last_number == 0:
                                        del Game._last_turn(game_name).stats[schema_name][item_index]

                            item_index = None
                            temp_string = ""
                            schema_name = None

                    yield Game.render_story(game_name)

                if count_tokens:
                    Game._(game_name).llm_model.num_tokens_from_text(model, Game._last_raw(game_name)[1])

                from PythonClasses.Game.FileManager import FileManager
                FileManager.save_history(game_name, game_name)

                stripped_string = '\n'.join([line for line in Game._last_display(game_name)[1].split('\n') if not line.startswith('--->')]).strip(" .\n")
                Game._(game_name).audio_file = Game._(game_name).audio.no_ssml(stripped_string,rate=audio_speed)

                if new_day:
                    logger.info("Starting a new day")
                    last_turn = Game._last_turn(game_name)

                    user_message = "{The day has ended. Begin the next day}"

                    new_day_json = {
                        "type": "normal",
                        "model": Game._history(game_name)[0].model,
                        "system_message": Game._history(game_name)[0].system_message,
                        "display": [None, None],
                        "raw": [user_message, None],
                        "stats": Game._stats(game_name).copy(),
                        "combat": [],
                        "execution": {},
                    }

                    Game._history(game_name).append(Turn(new_day_json))

            


            Game._(game_name).state = Game.AWAITING_USER

            yield Game.render_story(game_name)

        except Exception as e:
            Game._last_display(game_name)[0] = f"Try again! Encountered an exception: {e}"
            Game._last_display(game_name)[1] = None
            Game._last_raw(game_name)[0] = None
            Game._last_raw(game_name)[1] = None
            new_day = False
            Game._(game_name).state = Game.AWAITING_USER

            yield Game.render_story(game_name)