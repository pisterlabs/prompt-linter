import sys
from game import *
from place import *
from player import player
import os
import openai
import json
from pathlib import Path

def main():

    # Check if arguments have been passed to the script
    if len(sys.argv) < 1:
        # Print the first argument passed to the script
        quit()
    prompt = sys.argv[1]

    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.8,
        max_tokens=2049,
        top_p=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.0
    )

    print(response)

    with open("_sendGPTQuestion.output.txt", "w") as text_file:
        text_file.write("\""+prompt+"\"\n")
        text_file.write("---\n")
        text_file.write(response["choices"][0]["text"])
    
			
if __name__ == '__main__':
	main()
