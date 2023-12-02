from sys import *
from game import *
from place import *
from player import player
import os
import openai
import json
from pathlib import Path

def main():
    worldSrc = Path('../world.py').read_text()
    placeSrc = Path('../place.py').read_text()
    src = placeSrc + "\n\n" + worldSrc

    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Refer to the following Python classes which describe a game world.  Generate a map of the game world using ASCII art\n"+src,
        temperature=0.8,
        max_tokens=2049,
        top_p=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.0
    )

    with open("_generateMap.output.txt", "w") as text_file:
        text_file.write(response["choices"][0]["text"])
    
			
if __name__ == '__main__':
	main()
