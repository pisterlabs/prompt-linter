import logging
import typing as t
import json
import os
import openai
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv
logger = logging.getLogger(__name__)

history = ["You: How was your day?", "Identity name: My day was wonderful, Master."]
user_message = "What are your plans for tomorrow?"

load_dotenv()
owner_name = os.getenv('OWNER_NAME')

def read_identity_file(identity_path: str) -> Tuple[str, str, str, str, str]:
    """
    Read an identity file and return the character information.
    :param identity_path: The path to the identity JSON file
    :return: A tuple containing character name, persona, greeting, example dialogue, and world scenario
    """
    try:
        with open(Path(identity_path), "r", encoding="utf-8") as f:
            identity = json.load(f)

        char_name = identity["char_name"]
        char_persona = identity["char_persona"]
        char_greeting = identity["char_greeting"]
        example_dialogue = identity["example_dialogue"]
        world_scenario = identity["world_scenario"]

    except FileNotFoundError:
        logger.error(f"File {identity_path} not found.")
        return None, None, None, None, None
    except KeyError as e:
        logger.error(f"Key error: {e}. Please check the JSON file format.")
        return None, None, None, None, None

    return char_name, char_persona, char_greeting, example_dialogue, world_scenario


def build_prompt(char_name: str, char_persona: str, char_greeting: str, example_dialogue: str, world_scenario: str) -> str:
    """
    Build a prompt based on character information.
    :param char_name: The name of the character
    :param char_persona: The persona of the character
    :param char_greeting: The greeting used by the character
    :param example_dialogue: An example dialogue featuring the character
    :param world_scenario: A description of the world scenario
    :return: A generated prompt based on the given parameters
    """
    prompt = f"Role Play \n Character: {char_name}\n {char_persona} Greeting : {char_greeting}\n Example Dialogue: {example_dialogue}\n World Scenario: {world_scenario} <<start>>\n "

    return prompt

def send_prompt_to_openai(api_key: str, prompt: str) -> str:
    """
    Send the given prompt to OpenAI's GPT API and return the generated text.
    :param api_key: Your OpenAI API key
    :param prompt: The prompt text to send
    :return: The generated text from OpenAI's GPT
    """
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    generated_text = response.choices[0].text.strip()
    return generated_text

def createWaifuPrompt(json_file_path):
    identity_path = json_file_path
    char_name, char_persona, char_greeting, example_dialogue, world_scenario = read_identity_file(identity_path)

    if char_name and char_persona and char_greeting and example_dialogue and world_scenario:
        prompt = build_prompt(char_name, char_persona, char_greeting, example_dialogue, world_scenario)
        return prompt
        
        """
            #print("Generated prompt:", prompt)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OpenAI API key not found in environment variables.")
            return
        generated_text = send_prompt_to_openai(api_key, prompt)
        return generated_text
        #print("Generated prompt:", generated_text)
        """
        
"""
if __name__ == "__main__":
    file = "identity.json"
    createWaifuPrompt(file)
"""



