# File: pychat.py
# Description: A simple python chat program that uses the OpenAI API
#
# Usage: python pychat.py <filename> where filename is where to store the chat history
#
# The program keeps a history of the chat in memory and passes the whole chat back to the API for each new request
# This means that the AI can use the history to provide context for the response
# The program saves the history of the chat from each run in a json file
# You can change the system prompt and the temperature of the AI response using the system: and temp: commands
# See the help command for more information


import openai
import os
import datetime
import argparse
import json


default_data = {
    "role": {
        "name": "Advisor",
        "prompt": """
You are an expert advisor.
You provide advice with explanations, history, related topics, and examples.
If you aren't sure about something or don't know then say so.
Don't make things up.
Ask questions if you need more information to be able to advise.""",
        "temperature": 0.7
    },
    "chat": [],
    "transcript": "",
}
data = default_data.copy()

    
def display(message):
    print(message)
    data["transcript"] += message + "\n"
    save()


def save():
    with open(filename, "w") as f:
        json.dump(data, f)


def load(filename):
    data = default_data
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                data = json.load(f)
            except Exception as e:
                display(f"Error loading file {filename}: {e}")
                display("Starting a new chat file.")
    else:
        display(f"File {filename} does not exist. Creating a new chat file with this name.")        
    return data


openai.api_key = os.getenv("OPENAI_API_KEY")

now = datetime.datetime.now()
timestamp = now.strftime("%y%m%d")

parser = argparse.ArgumentParser(description="A python chat program that uses the OpenAI API")
parser.add_argument("filename", nargs="?", default=f"pychat_{timestamp}.json", help="the name of the chat file for load and save")
args = parser.parse_args()
filename = args.filename
if not filename.endswith(".json"):
    filename += ".json"
data = load(filename)

print(data["transcript"])
display(f"\n\nWelcome to pychat! Type 'help' for a list of commands.\n")
display(f"Current system prompt is:\n--------------------\n{data['role']['prompt']}\n--------------------\n")
display(f"Temperature is: {data['role']['temperature']}")
display(f"Chat history will be saved to {filename}")

ui_prompt = "pychat > "
while True:
    prompt = input(f"\n{ui_prompt}")
    if prompt.lower() in ["quit", "exit"]:
        display(f"Goodbye.\nYou can find the chat in the file {filename}\n")
        break
    if prompt.lower() in ["dump", "debug"]:
        print(json.dumps(data, indent=4, sort_keys=True))
        continue
    if prompt.startswith("system:"):
        if prompt[7:].strip() == "":
            display(f"Current system prompt is: {data['role']['prompt']}")
        else:
            data["role"]["prompt"] = prompt[7:].strip()
            display(f'System prompt set to {data["role"]["prompt"]}')
        continue
    if prompt.startswith("temp:"):
        try:
            data["role"]["temperature"] = float(prompt[5:].strip())
            if data["role"]["temperature"] < 0 or data["role"]["temperature"] > 2:
                raise Exception("Temperature must be between 0 and 2")
            display(f'Temperature set to {data["role"]["temperature"]}')
        except Exception as e:
            display(f'Error setting temperature: {e}')
        continue
    if prompt.startswith("load "):
        try:
            new_filename = prompt[5:].strip()
            if not new_filename.endswith(".json"):
                new_filename += ".json"
            data = load(new_filename)
            filename = new_filename
            display(f"Loaded chat history from {new_filename}, further chat will be saved here.")
        except Exception as e:
            display(f'Error attempting load: {e}')
        continue
    if prompt.startswith("saveas "):
        try:
            new_filename = prompt[7:].strip()
            if not new_filename.endswith(".json"):
                new_filename += ".json"
            # delete the file if it exists
            if os.path.exists(new_filename):
                # Check if they really want to overwrite the file
                if input(f"File {filename} already exists. Overwrite? (y/n) ").lower() != "y":
                    display(f"Saveas aborted. Continuing to save to {filename}.")
                    continue
                else:
                    try:
                        os.remove(new_filename)
                    except Exception as e:
                        display(f"Error deleting file {new_filename}: {e}")
                        display(f"Saveas aborted. Continuing to save to {filename}.")
                        continue
        except Exception as e:
            display(f'Error attempting saveas: {e}\nContinuing to save to {filename}.')
        filename = new_filename
        save()
        display(f"Saved and continuing to save chat to {filename}.")
        continue
    if prompt.lower() == "history":
        for item in data["chat"]:
            display(f"{item['role']}: {item['content']}")
        continue
    if prompt in ['help', 'man']:
        print(f"""
Chat being saved to: {filename}
System prompt: {data['role']['prompt']}
Temperature: {data['role']['temperature']}
Commands:
    quit or exit - exit the program
    dump or debug - display the json for the chat history
    system: - display the current system prompt
    system: <prompt> - set the system prompt (to define the role you want the AI to play)
    temp: <temperature> - set the temperature (to a floating point number between 0 and 2)
    history - display the chat history
    saveas <filename> - save the chat to a new .json file
    load <filename> - load a chat from a .json file
    help - display this help message
""")
        continue
    else:
        if prompt == "":
            # If the user enters a blank line, then we'll just repeat the last prompt
            # As long as the temperature isn't 0, it should give a different response
            prompt = data["chat"][-1]["content"]
            display(f"\n{ui_prompt} Repeating previous request: {prompt}")

        data["chat"].append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            #model="gpt-3.5-turbo-0613",
            model="gpt-4",
            messages=[{"role": "system", "content": data["role"]["prompt"]}] + data["chat"],
            temperature=data["role"]["temperature"],
        )

        system_response = response.choices[0]["message"]["content"]
        display(f"AI: {system_response}")
        data["chat"].append({"role": "assistant", "content": system_response})
        save()
