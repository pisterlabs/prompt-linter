import pathlib
import sys
from typing import cast, Any
import openai
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import entities
import pickle
import os
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import LLMChain, PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


class CustomCallbackHandler(StreamingStdOutCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        print(token, end="")
        print("|", end="")
        # sys.stdout.write(token)
        # sys.stdout.flush()


class api_keyholder:
    def __init__(self):
        self.key = ""

    def __call__(self):
        return self.key

    # set the key
    def set_key(self, key):
        openai.api_key = key
        os.environ['OPENAI_API_KEY'] = key
        self.key = key


api_key = api_keyholder()
if pathlib.Path("api.txt").exists():
    api_key.set_key(pathlib.Path("api.txt").read_text().strip(""))


def convert_history_list_to_string(item_list: list[str]):
    _result = ""
    for item in item_list:
        _result += f"> {item}\n"
    _result = _result.strip()
    return _result


def convert_item_list_to_string(item_list: list[str]):
    _result = ""
    for item in item_list:
        _result += f"{item}\n"
    _result = _result.strip()
    return _result


def extract_result_from_quotes(result: str):
    number_of_quotes = result.count('"')
    if number_of_quotes < 2:
        return result
    first_quote = result.find('"')
    last_quote = result.rfind('"')
    result = result[first_quote + 1:last_quote]
    return result


def convert_tense_fast(action: str):
    fast_model = OpenAI(model_name='text-davinci-003')
    fast_template = """Given a player's move, which may use language like "I will" or "I do this", 
convert the player's move so that it uses language like "I try to" or "I attempt to".

# PLAYER'S MOVE:
{action}

# NEW VERSION:"""
    fast_prompt = PromptTemplate(template=fast_template, input_variables=["action"])
    llm_chain = LLMChain(
        prompt=fast_prompt,
        llm=fast_model
    )
    result = llm_chain.run(action)
    return result


def sanitize_action(player, _action: str):
    # _action = extract_first_step(_action)
    _action = convert_tense_fast(_action)
    return _action.strip()


default_player_context_template = """# PLAYER's CONTEXT:

### PLAYER's CHARACTER DESCRIPTION:

{player_character}

### WORLD DESCRIPTION:

{world}

### PLAYER'S LOCATION:

{player_location}

### PLAYER'S INVENTORY:

{player_inventory}"""

gpt_4_version = True


def get_dungeon_master_thoughts(action: str,
                                player: entities.Player,
                                dungeon_master: entities.DungeonMaster):
    # chat_bot = ChatOpenAI(temperature=0, model_name="gpt-4")
    chat_bot = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    if gpt_4_version:
        system_template = """
You are a mediator in a dungeons and dragons game.
You will be given a player's move (and context), and you are to use the context
to come up with the dungeon master's thoughts about the player's move.
Think about whether it the move is possible currently in the story, how likely the move is to succeed, and whether it is fair.
Write your thoughts down in a single sentence. Make it extremely short.
If the move is unfair or difficult for the player, state why.
If the move is not inline with the theme of the world, state why.
Mention any pro or any con of the move.
Keep your thoughts short and very concise.
"""
    else:
        system_template = """
You are a mediator in a dungeons and dragons game.
You will be given a player's move (and context), and you are to use the context
to come up with the dungeon master's thoughts about the player's move.
The move MUST be a single small action that doesn't progress the story much - don't let the player cheat.
Consider whether you will allow them to progress through the story with this move. Letting the player progress sometimes makes the game fun.
Think about whether it the move is possible currently in the story, how likely the move is to succeed, and whether it is fair.
Write your thoughts down in a single sentence. Make it extremely short.
The quest campaign story is hidden from the player, do not reveal future events, or any information or secrets that have not yet been given to the player.
"""
    system_context = default_player_context_template

    system_context_2 = """### PLAYER'S ACTION HISTORY:

{action_history}

### SECRET QUEST CAMPAIGN STORY (hidden from the player):

{story}"""

    user_template = """# PLAYER'S MOVE:

{players_move}

# THOUGHTS:"""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    system_context_prompt = SystemMessagePromptTemplate.from_template(system_context)
    system_context_prompt_2 = SystemMessagePromptTemplate.from_template(system_context_2)
    user_message_prompt = HumanMessagePromptTemplate.from_template(user_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, system_context_prompt, system_context_prompt_2, user_message_prompt])

    chain = LLMChain(llm=chat_bot, prompt=chat_prompt)
    result = chain.run(
        players_move=action,
        player_character=player.description,
        player_location=player.location,
        player_inventory=convert_item_list_to_string(player.items),
        world=dungeon_master.world_description,
        action_history=convert_history_list_to_string(dungeon_master.player_summaries),
        story=dungeon_master.quest_story
    )
    return result


def extract_yes_no(text: str, default: bool = True) -> bool:
    text = text.lower()
    if "[YES]" in text and "[NO]" in text:
        return default
    if "[YES]" in text:
        return True
    if "[NO]" in text:
        return False
    text = text.replace("\t", " ")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("  ", " ")
    text = text.replace(".", " ")
    text = text.replace(",", " ")
    text = text.replace("?", " ")
    text = text.replace("!", " ")
    text = text.replace(":", " ")
    text = text.replace(";", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("[", " ")
    text = text.replace("]", " ")
    text = text.replace("{", " ")
    text = text.replace("}", " ")
    text = text.replace("<", " ")
    text = text.replace(">", " ")
    text = text.replace("\"", " ")
    text_list = text.split(" ")
    for word in text_list:
        word = word.strip()
        if word == "yes":
            return True
        if word == "no":
            return False
    return default


def get_likely_outcome(
        player: entities.Player,
        player_sanitized_action: str,
        player_thoughts: str,
        dungeon_master: entities.DungeonMaster,
):
    chat_bot = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    if gpt_4_version:
        system_template = """
You are the dungeon master in a dungeons and dragons game.
You will be given the action of the player of the game and you will need to state the likely outcome of the action, given the thoughts and the context.
Generate the likely action directly from the thoughts.
Consider whether the move is even possible currently in the story, how likely the move is to succeed, and whether it is fair.
Consider whether you will allow them to progress through the story with this move. Letting the player progress sometimes makes the game fun.
Make sure the outcome is written concisely, keeping it very short.
The quest campaign story is hidden from the player, do not reveal future events, or any information or secrets that have not yet been given to the player.
"""
    else:
        system_template = """
You are the dungeon master in a dungeons and dragons game.
You will be given the action of the player of the game and you will need to state the likely outcome of the action, given the thoughts and the context.
Generate the likely action directly from the thoughts.
Consider whether the move is even possible currently in the story, how likely the move is to succeed, and whether it is fair.
Consider whether you will allow them to progress through the story with this move. Letting the player progress sometimes makes the game fun.
Make sure the outcome is written concisely, keeping it very short.
The quest campaign story is hidden from the player, do not reveal future events, or any information or secrets that have not yet been given to the player.
"""
    system_context_1 = default_player_context_template

    system_context_2 = """### PLAYER'S ACTION HISTORY:

{main_history}

### SECRET QUEST CAMPAIGN STORY (hidden from the player):

{story}"""

    user_template = """# PLAYER'S ACTION:

{player_action}

# YOUR THOUGHTS ON THE PLAYER'S ACTION:

{player_action_thoughts}

# LIKELY OUTCOME OF PLAYER'S ACTION:"""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    system_context_prompt_1 = SystemMessagePromptTemplate.from_template(system_context_1)
    system_context_prompt_2 = SystemMessagePromptTemplate.from_template(system_context_2)
    user_message_prompt = HumanMessagePromptTemplate.from_template(user_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, system_context_prompt_1, system_context_prompt_2, user_message_prompt])

    chain = LLMChain(llm=chat_bot, prompt=chat_prompt)

    result = chain.run(
        player_action=player_sanitized_action,
        player_action_thoughts=player_thoughts,
        player_inventory=convert_item_list_to_string(player.items),
        player_character=player.description,
        player_location=player.location,
        world=dungeon_master.world_description,
        main_history=convert_history_list_to_string(dungeon_master.player_summaries),
        story=dungeon_master.quest_story
    )
    return result


def new_token():
    print("new token")


def get_gpt_4_dungeon_master_outcome(
        gpt_4_api_key: str,
        DMTokenCallbackHandler,
        player: entities.Player,
        player_sanitized_action: str,
        player_thoughts: str,
        # player_likely_outcome: str,
        dungeon_master: entities.DungeonMaster,
):
    original_key = api_key.key
    api_key.set_key(gpt_4_api_key)
    chat_bot = ChatOpenAI(temperature=0, model_name="gpt-4")
    #chat_bot = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    #     system_template = """
    # You will be given the action of a player in a Dungeons and Dragons game.
    # You are to write a short description of the immediate outcome of their action.
    # Do not progress any further than the likely outcome. Do not add anything to the likely outcome.
    # Use the likely outcome exactly as they are given."""

    system_template = """
You are the dungeon master of a singleplayer text-adventure Dungeons and Dragons game. The game should be challenging. Stupid choices
should be punished and should have consequences.
The player has just taken their action, and the outcome is given to you. Write a short single paragraph of the immediate outcome of their action.
If the player is not doing an action that is in-line with the story, they should be allowed to go ahead with their action, but the outcome you write shouldn't
progress the story.
The outcome should contain MULTIPLE story hooks in the paragraph (embedded different sub-stories that are happening in the background).
Once you have written this short single paragraph, then give a very short single sentence description of what is around the player,
prioritising mentioning any people, buildings, or any other things of interest, this is because
it is a text-adventure game, and the player can't see.
Write it like you are telling the player what happened to them., using language like "you" and "your".
Use imaginative and creative language with lots of enthusiasm.
Don't tell the player what they should do next, simply ask, "what do you do next?".
The quest campaign story is hidden from the player, do not reveal future events, or any information or secrets that have not yet been given to the player."""

    context_template_player = default_player_context_template

    system_context_2 = """### HISTORY OF THE GAME SO FAR:

{player_action_history}

### SECRET QUEST CAMPAIGN STORY (hidden from the player):

{story}"""

    user_combined_template = """
# PLAYER'S ACTION:

{player_action}

### YOUR THOUGHTS ABOUT THE PLAYER'S ACTION:

{player_thoughts}

# DUNGEON MASTER'S RESPONSE:"""

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    system_message_prompt_2 = SystemMessagePromptTemplate.from_template(context_template_player)
    system_message_prompt_3 = SystemMessagePromptTemplate.from_template(system_context_2)
    user_message_prompt = HumanMessagePromptTemplate.from_template(user_combined_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                                    system_message_prompt_2,
                                                    system_message_prompt_3,
                                                    user_message_prompt])

    chain = LLMChain(llm=chat_bot, prompt=chat_prompt)
    result = chain.run(
        world=dungeon_master.world_description,
        player_inventory=convert_item_list_to_string(player.items),
        player_character=player.description,
        player_location=player.location,
        player_action=player_sanitized_action,
        player_thoughts=player_thoughts,
        player_action_history=convert_history_list_to_string(dungeon_master.player_summaries),
        story=dungeon_master.quest_story
    )
    api_key.set_key(original_key)
    return result


def get_dungeon_master_outcome(
        player: entities.Player,
        player_sanitized_action: str,
        player_thoughts: str,
        player_likely_outcome: str,
        dungeon_master: entities.DungeonMaster,
):
    chat_bot = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    #     system_template = """
    # You will be given the action of a player in a Dungeons and Dragons game.
    # You are to write a short description of the immediate outcome of their action.
    # Do not progress any further than the likely outcome. Do not add anything to the likely outcome.
    # Use the likely outcome exactly as they are given."""

    system_template = """
You are the dungeon master of a Dungeons and Dragons game.
The player has just taken their action, and the outcome is given to you. However, the language used isn't correct.
You are to correct the language without changing the meaning of the outcome.
You are to direct the outcome at the player, using language like "you" and "your". Use imaginative and creative language with lots of enthusiasm.
Write it like you are telling the player what happened to them.
The quest campaign story is hidden from the player, do not reveal future events, or any information or secrets that have not yet been given to the player."""

    context_template_player = default_player_context_template

    system_context_2 = """### PLAYER'S ACTION HISTORY:

{player_action_history}

### SECRET QUEST CAMPAIGN STORY (hidden from the player):

{story}"""

    user_combined_template = """
# PLAYER'S ACTION:
    
{player_action}

### YOUR THOUGHTS ABOUT THE PLAYER'S ACTION:

{player_thoughts}

### THE OUTCOME OF PLAYER'S ACTION:

{player_likely_outcome}

# REWORDED OUTCOME OF PLAYER'S ACTION:"""

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    system_message_prompt_2 = SystemMessagePromptTemplate.from_template(context_template_player)
    system_message_prompt_3 = SystemMessagePromptTemplate.from_template(system_context_2)
    user_message_prompt = HumanMessagePromptTemplate.from_template(user_combined_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                                    system_message_prompt_2,
                                                    system_message_prompt_3,
                                                    user_message_prompt])

    chain = LLMChain(llm=chat_bot, prompt=chat_prompt)
    result = chain.run(
        world=dungeon_master.world_description,
        player_inventory=convert_item_list_to_string(player.items),
        player_character=player.description,
        player_location=player.location,
        player_action=player_sanitized_action,
        player_thoughts=player_thoughts,
        player_likely_outcome=player_likely_outcome,
        player_action_history=convert_history_list_to_string(dungeon_master.player_summaries),
        story=dungeon_master.quest_story
    )
    return result


def determine_new_location(
        player: entities.Player,
        player_sanitized_action: str,
        outcome: str,
        dungeon_master: entities.DungeonMaster,
) -> str:
    chat_bot = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    #     system_template = """
    # You will be given the action of a player in a Dungeons and Dragons game.
    # You are to write a short description of the immediate outcome of their action.
    # Do not progress any further than the likely outcome. Do not add anything to the likely outcome.
    # Use the likely outcome exactly as they are given."""

    system_template = """You are a location determining machine. Given an old location, world context, and player action, you are to determine the location of the player during/at the end of their action.
The location may be the same as before. Use the context to help you determine the location. The location should be stated in a single concise sentence. Write the location in quotes. Don't say "You are still" or "You are now". Say: "You are"
This is so that the full location can be displayed to the player. It is important that the player knows where they are, even if they leave the game for a while and come back later, there should be enough information for them to know where they are."""

    context_template_player = """# WORLD CONTEXT:

### WORLD DESCRIPTION:

{world}"""

    system_context_2 = """"""

    user_combined_template = """### STORY HISTORY:

"{player_action_history}"

# PLAYER'S PREVIOUS LOCATION:

"{player_location}"
    
# PLAYER'S LATEST ACTION:

"{player_action}"

# THE OUTCOME GIVEN TO THE PLAYER:

"{outcome}"

# THE PLAYER'S NEW LOCATION:"""

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    system_message_prompt_2 = SystemMessagePromptTemplate.from_template(context_template_player)
    system_message_prompt_3 = SystemMessagePromptTemplate.from_template(system_context_2)
    user_message_prompt = HumanMessagePromptTemplate.from_template(user_combined_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                                    system_message_prompt_2,
                                                    system_message_prompt_3,
                                                    user_message_prompt])

    chain = LLMChain(llm=chat_bot, prompt=chat_prompt)
    result = chain.run(
        world=dungeon_master.world_description,
        player_action_history=convert_history_list_to_string(dungeon_master.player_summaries),
        player_location=player.location,
        player_action=player_sanitized_action,
        outcome=outcome,
    )
    return extract_result_from_quotes(result)


def summarise_event(
        action: str,
        outcome: str,
):
    chat_bot = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    system_template = """
        """
    user_template = """Given the input action and input action outcome, you are to summarise the event, keeping ALL important information, but using very few words and concise language.
Also, make sure that it is directed towards the player, using words like "you" and "your".
Write the output text in quotes.
# INPUT ACTION:

{action}

# INPUT ACTION OUTCOME:

{outcome}

# SUMMARISED OUTPUT:"""

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    user_message_prompt = HumanMessagePromptTemplate.from_template(user_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                                    user_message_prompt])

    chain = LLMChain(llm=chat_bot, prompt=chat_prompt)
    result = chain.run(
        action=action,
        outcome=outcome,
    )
    return extract_result_from_quotes(result)


def write_visual_description(
        player: entities.Player,
        dungeon_master: entities.DungeonMaster,
        event_summary: str,
):
    chat_bot = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    system_template = """
You will be given a scenario with lots of information, along with the latest EVENT SUMMARY.
You are to convert the latest event (using the context too) into a single sentence of what the scene looks like during the event.
The visual prompt must describe VISUALLY what the scene looks like. Make sure to include what the foreground and the background looks like. Also include the setting, such as "fantasy" or "medieval".
Make sure to include what the location looks like.
Include ONLY the most crucial details that make up what the particular event looks like to an observer."""

    user_template = """
# PLAYER'S CHARACTER DESCRIPTION:

{player_character}

# WORLD DESCRIPTION:

{world}

# PLAYER'S LOCATION:

{player_location}

# EVENT SUMMARY:

{event_summary}

# EXACT VISUAL DESCRIPTION:"""

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    user_message_prompt = HumanMessagePromptTemplate.from_template(user_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                                    user_message_prompt])

    chain = LLMChain(llm=chat_bot, prompt=chat_prompt)
    result = chain.run(
        world=dungeon_master.world_description,
        player_inventory=convert_item_list_to_string(player.items),
        player_character=player.description,
        player_location=player.location,
        event_summary=event_summary,
    )
    return result


def write_scenario_prompt(
        scenario: str,
):
    chat_bot = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    system_template = """
You are a machine that generates a visual prompt that will be turned into a painting, based upon a given scenario.
Include ONLY the most crucial details that make up what the particular event looks like to an observer. Follow a similar style to the examples given.
Make sure it is a very short single sentence.
Good prompt examples are as follows:

A painting of a warrior with a shield on his back and a sword in his hand, standing in front of a cave entrance. Mountains in the background. Fantasy. Highly detailed, Artstation, award winning.

A zoomed out painting of a siege of a medieval castle in winter while two great armies face each other fighting below and catapults throwing stones at the castle destroying its stone walls. fantasy, atmospheric, detailed.

A painting of a young man standing inside of a shop, browsing its wares. The shop is filled with various items, including weapons, armor, and potions. The shopkeeper is standing behind the counter, watching the young man. fantasy, sharp high quality, cinematic.

A painting of a beautiful matte painting of glass forest, a single figure walking through the middle of it with a battle axe on his back, cinematic, dynamic lighting, concept art, realistic, realism, colorful.

A closeup painting of an old wise villager, highly detailed face, depth of field, moody light, golden hour, fantasy, centered, extremely detailed, award winning painting.

A portrait painting of a butcher in a medieval village, holding a knife in his hand, with a dead pig hanging from a hook behind him. fantasy, sharp, high quality, extremely detailed, award winning painting.
"""

    user_template = """
# DESCRIPTION OF THE SCENARIO:

{scenario}
    
# VISUAL PROMPT:"""

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    user_message_prompt = HumanMessagePromptTemplate.from_template(user_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                                    user_message_prompt])

    chain = LLMChain(llm=chat_bot, prompt=chat_prompt)
    result = chain.run(
        scenario=scenario,
    )
    return result


def generate_image(prompt: str):
    forced_additional_prompt = "fantasy, desaturated"
    prompt = prompt.strip()
    if prompt[-1] == ".":
        prompt = prompt[:-1]
    if prompt[-1] == ",":
        prompt = prompt[:-1]
    prompt = prompt + ". " + forced_additional_prompt
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="512x512",
    )
    image_url = response['data'][0]['url']
    return image_url
