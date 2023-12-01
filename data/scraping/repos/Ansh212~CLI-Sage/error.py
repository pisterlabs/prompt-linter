import openai
import subprocess
import os
from typing import Dict, Any
import sys
import signal
import time
import config
from colorama import Fore, Back, Style

color_code = "1;32"

def signal_handler(signal, frame):
    print()
    print(Style.BRIGHT + Fore.YELLOW + "Bye" + Style.RESET_ALL)
    sys.exit()


# Set your OpenAI API key here
openai.api_key = config.api_key

def chat_with_gpt(prompt: str , role: str) -> str:

    role_call=""

    if(role == "d"):
        role_call = "You are an error message debugger. Explain what the error is and why it is occurring .Do not give the corrected code"
    else:
        role_call = "You are code modifier , you will be given the errors , steps to solve the errors and the code . Give the modified code only"

    response: Dict[str, Any] = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the GPT-3.5 Turbo engine
        messages=[
            {"role": "system", "content": role_call},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )

    generated_text = ""  # Initialize generated_text here

    try:
        # Print the "Sage:" prompt at the beginning
        print(Style.BRIGHT + "Sage:", end=' ', flush=True)

        for chunk in response:
            delta_content = chunk['choices'][0]['delta'].get('content', '')
            if delta_content:
                # Print subsequent responses on the same line
                for i in delta_content:
                    generated_text += i
                    colortext = color_print(i, color_code)
                    print(Style.BRIGHT + Fore.LIGHTCYAN_EX + i + Style.RESET_ALL, end='', flush=True)
                    time.sleep(0.02)
    except Exception as e:
        print(f"Error: {e}")

    print()  # Move to the next line after streaming the response
    return generated_text

def color_print(text, color_code):
    formatted_text = f"\033[{color_code}m {text} \033[0m"
    return formatted_text 

def reviewFile(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()

    prompt = "Find errors in the code"
    user_message = f"{prompt}/n{file_content}"

    error_sol = chat_with_gpt(user_message , "d")

    doModify = doModify = input(Style.BRIGHT + Fore.YELLOW + ">>> Do you want to AI to modify the file to solve the errors? [y/n] :" + Style.RESET_ALL)


    if(doModify == "n"):
        return
    mod_code = chat_with_gpt(error_sol + file_content , "m")

    with open(file_path , 'w') as file:
        file.write(mod_code);

    return

def handle_command(com_string):

    split_string = com_string.split()

    if split_string[0] == "reviewFile":
        reviewFile(split_string[1])

if __name__ == "__main__":
    prev_res = ""
    response = ""

    print(Style.BRIGHT + Fore.YELLOW + "Press CTRL-C to exit" + Style.RESET_ALL)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            user_input = input(">>> ")

            if user_input.startswith('/'):
                handle_command(user_input[1:])
                continue

            try:
                result = subprocess.run(user_input, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print("Command executed successfully")
                print("Output:", result.stdout)
            except subprocess.CalledProcessError as e:
                print("Error occurred during command execution")
                print("Exit Code:", e.returncode)
                print("Error Output:", e.stderr)
                response = chat_with_gpt(e.stderr , "d")
                color_print(response, color_code)

            if user_input.lower() in ['quit', 'exit', 'bye', 'q', 'x', 'b']:
                break

            if prev_res != "":
                prev_res = "This was the previous command " + prev_res

            prev_res = response

    except KeyboardInterrupt:
        print("Okay")
