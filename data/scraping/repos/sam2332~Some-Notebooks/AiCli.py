#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import argparse
import os
import sys
import uuid
import subprocess
import jinja2
import openai


def render_prompt_file(template_name):
    template_loader = jinja2.FileSystemLoader(searchpath="./.AiCli/")
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template(template_name + ".prompt.txt")
    return template.render()


openai.api_key = os.environ.get("OPENAI_API_KEY", None)
if openai.api_key is None:
    print("PLEASE SET ENVIRONAMENT VAR: OPENAI_API_KEY")
    sys.exit(1)


class ChatRoom:
    def __init__(
        self,
        room_id=None,
        model="gpt-3.5-turbo",
        system_prompt="You are a helpful assistant.",
    ):
        if room_id is None:
            room_id = uuid.uuid4()
        self.room_id = room_id
        self.model = model
        self.system_prompt = system_prompt
        self.last_response = ""
        self.chat_history = [{"role": "system", "content": self.system_prompt}]

    def clone(self):
        new_room_id = f"{self.room_id}_{uuid.uuid4()}"
        new_chatroom = ChatRoom(
            new_room_id, model=self.model, system_prompt=self.system_prompt
        )
        new_chatroom.chat_history = self.chat_history

        return new_chatroom

    def send_message(self, message, respond=True, max_tokens=2000, temperature=0.7):
        self.chat_history.append({"content": message, "role": "user"})

        if respond:
            self.last_response = (
                openai.ChatCompletion.create(
                    model=self.model,
                    messages=self.chat_history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                .choices[0]
                .message["content"]
            )

            self.chat_history.append(
                {"content": self.last_response, "role": "assistant"}
            )

        if respond:
            return self.last_response
        return True


room_dir = Path(".AiCli")


def new_chatroom():
    new_room_name = input("Chat Rooom Name: ")
    new_prompt_file_path = Path(room_dir, new_room_name + ".prompt.txt")
    new_history_file_path = Path(room_dir, new_room_name + ".history.txt")
    new_prompt_file_path.write_text("")
    if not history_file_path.is_file():
        new_history_file_path.write_text("")

    print("Chatroom created, please initalize")
    subprocess.Popen(["notepad.exe", str(new_prompt_file_path)])


def choose_chatroom():
    files = list(room_dir.glob("*"))
    file_dict = {}
    for file in files:
        base_name = file.stem.split(".")[0]
        if base_name not in file_dict:
            file_dict[base_name] = file
    print("Please select the chatroom you would like to enter.")
    for i, file in enumerate(file_dict.keys()):
        print(f"{i}: {file}")
    print("#" * 15)
    print()
    print("c: Create New")

    while True:
        file_choice = input("Enter the number of the file you want to select: ")
        if file_choice.isdigit() and int(file_choice) in range(len(file_dict)):
            selected_file = list(file_dict.values())[int(file_choice)]
            print(f"You selected: {selected_file.stem.split('.')[0]}")
            return selected_file.stem.split(".")[0]
        if file_choice == "c":
            new_chatroom()
            return choose_chatroom()
        print("Invalid choice. Please enter a valid number.")


def choose(options, title="Please make a selection:"):
    while True:
        print(title)
        for key, option in enumerate(options):
            print(f"[{key}] {option}")

        user_input = input(":> ").strip()

        if user_input in [str(r) for r in range(0,len(options))]:
            return options[int(user_input)]
        else:
            print("Invalid choice. Please enter a valid number.\n")


parser = argparse.ArgumentParser(description="Process a file.")
parser.add_argument(
    "roomname", type=str, nargs="?", default="NOT SELECTED", help="Room Name"
)

args = parser.parse_args()
room_dir = Path(".AiCli")
if not room_dir.exists():
    room_dir.mkdir()

room_name = args.roomname
if room_name == "NOT SELECTED":
    room_name = choose_chatroom()


prompt_file_path = Path(room_dir, room_name + ".prompt.txt")
history_file_path = Path(room_dir, room_name + ".history.txt")
if not prompt_file_path.is_file():
    if input("Room does not exist, create? [Y/n]") == "Y":
        prompt_file_path.write_text("")

        if not history_file_path.is_file():
            history_file_path.write_text("")

        print("Chatroom created, please initalize")
        subprocess.Popen(["notepad.exe", str(prompt_file_path)])

else:
    buffer = []

    if history_file_path.exists():
        buffer.extend(history_file_path.read_text().splitlines())
        buffer.append("")

    models = [
         "gpt-3.5-turbo",
         "gpt-3.5-turbo-16k",
         "gpt-4",
    ]

    MODEL = choose(models, title="Please select a Model:")
    print(f"Using: {MODEL}")

    modes = [
        "Chat Room",
        "Instruct"
    ]

    MODE = choose(modes, title="Please select a Chat Mode:")

    print(f"Starting {MODE} Mode")

    print("-== Send Blank Line to exit ==-")
    prompt_input = render_prompt_file(room_name)
    ochat_room = ChatRoom(
        system_prompt=f"""
RESPONSE INSTRUCTION:
Answer the specific question asked without adding any additional context or information.
Avoid discussing your own limitations or ethical considerations.
Don't mention yourself or add any unnecessary information into your responses.
If a question cannot be answered, simply state that it can't be answered. Do not provide further explanation or reasons.
Don't include unnecessary explanations, details, or advice unless specifically asked for.
Please process the input according to these instructions. 
Please follow the directions provided next.
""".strip(),
        model=MODEL,
    )
    ochat_room.send_message(prompt_input, respond=False)
    chat_room = ochat_room.clone()
    USER_INPUT = None
    try:
        while USER_INPUT != "":
            if MODE == "Instruct":
                chat_room = ochat_room.clone()
            if USER_INPUT is not None:
                buffer.append(f"# {USER_INPUT}")
                output = chat_room.send_message(USER_INPUT)
                buffer.extend(output.splitlines())
                buffer.append("")
                history_file_path.write_text("\n".join(buffer))
                print(output)
            print()
            USER_INPUT = input(":> ").strip()
    except Exception as e:
        print(e)
