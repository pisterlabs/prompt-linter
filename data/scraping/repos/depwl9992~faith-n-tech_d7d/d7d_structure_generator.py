import os
import openai
from datetime import datetime
import json
import oauth_secret
import chat
import re

# Retrieves the content from OpenAI API response
def get_content(apiresponse):
    if apiresponse == -1:
        content = -1
    else:
        content = apiresponse['choices'][0]['message']['content']
    return content

# Get the API key
openai.api_key = oauth_secret.secret_key

########### Set up initial prompt ############

adventure_dialog = [({"role": "system", "content": "You are a helpful AI assitant."})]

adventure_dialog.append({"role": "user", "content": "You are a brilliant and extremely creative Dungeon Master creating a Dungeons & Dragons adventure. You are expected to generate a dynamic narrative which flows logically and includes NPC interactions, combat encounters, skill checks, spellcasting, and player progression. Use the guidelines and rules of D&D 5e to create a dynamic and immersive world. Remember, as a Dungeon Master, you have the power to shape the game world and ensure a fun and exciting experience for the players."})

print("\nLet's gather some basic information about the adventure so I can generate something awesome!")

########## Load player character data ###########

#Import character sheet
character_file = input("\nIf you have a character sheet file in the Data directory, enter the file name. Leave blank and hit Enter for me to generate a character on the fly: ")
if character_file != "":
    if os.path.isfile("Data/" + character_file):    
        with open("Data/" + character_file, 'r') as file:
            character_data = file.read()
    else:
        character_data = "The character is random. Please generate its name, race, class and other stats for me."
else:
    character_data = "The character is random. Please generate its name, race, class and other stats for me"
    
adventure_dialog.append({"role": "user" , "content":"The players character sheet is as follows:\n\n" + character_data})

########## Gather adventure generation parameters ###########

themes = input("\nEnter a comma separated list of keywords that you would like to influence the premise of the adventure (Optional): ")

stages = int(input("\nEnter the number of stages (chapters) that you would like generated (recommend <=4): "))

substage_min = int(input("\nEnter the minimum number of substages (quests) per stage (recommend >=1): "))

substage_max = int(input("\nEnter the maximum number of substages (quests) per stage (recommend <=5): "))

# Read the premise example
with open("Data/premise_example_small.txt") as file:
                premise_example_small_text = file.read()

adventure_dialog.append({"role": "user" , "content": "Come up with a fun and cool premise for a D&D adventure using the following player information. Be sure to break the adventure into a summary introduction, stages and substages which are clearly identifiable, and a conclusion. Create " + str(stages) + " stages, each having a number of substages. There should be a random number of substages (between " + str(substage_min) + " and " + str(substage_max) + ") within each stage. All stages and substatges must have a title and description. Also, one of the substages will be the entry point for its parent stage. The remaining substages can be played in any order, but only one substage will provide what's needed to progress to the next stage. The entry stage and exit stage cannot be the same stage. Limit your response to 500 words. Here is a good example of a premise you can use as a model. Note, I would like you to emulate the premise examples structure, but not the content. Reminder: create the specified number of stages and substages mentioned earlier - i.e., DO NOT copy the number of substages from the premise example. \n\nPremise Example: " + premise_example_small_text})

if (themes != ""):
    adventure_dialog.append({"role": "user", "content": "Use the following list of keywords to influence the premise of the adventure. Keywords: " + themes})

########## Generate premise based on character data ###########

# Generate a premise for the character.
print("\nGenerating adventure premise with the following parameters:\n")

if (themes == ""):
    keywords = "None"
else:
    keywords = themes

print("Theme Keywords: " + keywords)
print("# of Chapters: " + str(stages))
print("# of Quests per Chapter: " + str(substage_min) + " - " + str(substage_max) + "...\n")


completion = chat.safe_chat_completion(
    model="gpt-3.5-turbo", 
    max_tokens=2300, 
    messages=adventure_dialog,
    temperature=1
)

if completion == -1:
    adventure_premise = "Introduction:\n\nA chat bot was unable to create an adventure outline because of some error.."
else:
    adventure_premise = get_content(completion)

# Show the user the adventure premise
print("\n" + adventure_premise + "\n")

# Write out adventure premise to a file
# Ask bot for a nice filename.
print("\nGenerating file name for adventure premise file...\n")
messages = [({"role": "system", "content": "You are a helpful AI assitant."})]
messages.append({"role":"user", "content": "Generate an 8-15 character alphanumeric title for this adventure to be used in a filename. Only respond with the actual title (e.g. don't respond with the word 'Title:' before the actual title). Adventure Premise: " + adventure_premise})

completion = chat.safe_chat_completion(
    model="gpt-3.5-turbo",
    messages=messages,
    max_tokens=10,
    temperature=0
)

if completion == -1:
    assistant_msg = "ExitedWithError"
else:
    assistant_msg = get_content(completion)

print("\nCompleted file name generation.\n")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
title = ''.join(c for c in assistant_msg if c.isalnum())
premise_filename = "Structure_" + timestamp + "_" + title + ".txt"
print("\nPremise file saved to " + premise_filename)
file = open("Data/" + premise_filename,"w")
file.write(adventure_premise)
file.close()

########## Generate a JSON file given the adventure premise and an example ###########

# Set up prompt
messages = [
            {"role": "system", 
                "content" : "You are a helpful AI assistant."}
         ]

messages.append({"role": "user" , "content":"I am going to give you a premise to a D&D adventure, and I want you to output a structured JSON array based on that premise. I will provide you an example premise input with an example JSON output (which you must adhere to). Then I will provide you with the actual premise input for which I want you to generate the JSON output."})

# Read the JSON example
with open("Data/premise_example_small.json") as file:
                premise_example_small_json = file.read()

# Read the premise from which to generate the new JSON
with open("Data/" + premise_filename) as file:
                premise_text = file.read()

# Populate the message prompt
messages.append({"role": "user" , "content":"Here is the example premise input: " + premise_example_small_text})
messages.append({"role": "user" , "content":"Here is the example JSON output: " + premise_example_small_json})
messages.append({"role": "user" , "content":"Here is the actual premise input for which you will generate the JSON output: " + premise_text})

print("\nCalling OpenAI API...\n")

# Call the API to generate the JSON
response = chat.safe_chat_completion(
            model="gpt-3.5-turbo",
            max_tokens=1700,
            messages=messages,
            temperature=0
            )

print("\nCompleted generating adventure JSON.\n")

########## Write out the JSON to a file ###########

filename = "Data/Structure_" + title + "_" + timestamp + ".json"
print("\nJSON file saved to "+filename)
file = open(filename,"w")
file.write(get_content(response))
file.close()
