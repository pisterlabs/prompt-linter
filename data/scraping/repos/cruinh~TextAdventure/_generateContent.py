from sys import *
from game import *
from place import *
from player import player
import os
import openai
import json
from pathlib import Path

def main():
    src = Path('../places/atrium.py').read_text()

    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Create a variation of this python class to represent a teen girl's bedroom. Include one exit leading to the hallway. \n"+src,
        temperature=0.8,
        max_tokens=2049,
        top_p=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.0
    )

    print(response)

    with open("../places/_generated.py", "w") as text_file:
        text_file.write(response["choices"][0]["text"])
    
			
if __name__ == '__main__':
	main()
