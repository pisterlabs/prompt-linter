import openai
import os
import subprocess
import colorama
from colorama import Fore, Style  # Import Fore and Style classes for color and style

# Initialize colorama
colorama.init()

#
def banner():
    
    print(Fore.WHITE + "")
    print(Fore.WHITE +"")
    print(Fore.RED +"   -:::::':::::::..    ...    ::::: ::::::::::::.-:.     ::-.  .,-:::::/   :::.       .,-:::::  :::  .   ")
    print(Fore.YELLOW +" ;;;'''' ;;;;``;;;;   ;;     ;;;; ;;;;;;;;;'''' ';;.   ;;;;',;;-'````'    ;;`;;    ,;;;'````'  ;;; .;;,.")
    print(Fore.YELLOW +" [[[,,==  [[[,/[[['  [['     [[[[ [    [[[        '[[,[[['  [[[   [[[[[[  [[ '[[,  [[[         [[[[[/'  ")
    print(Fore.WHITE +" `$$$'``  $$$$$$c    $$      $$$$$     |$$          c$$      $$c.    '$$  $$cc$$$c $$$        _$$$$,    ")
    print(Fore.WHITE +"  888     888b  88bo,886   .d88888      88,       ,8P'`       Y8bo,,,o88  888   888,`88bo,__,o,'888'88o, ")
    print(Fore.WHITE +"  MM,    MMMM   'W'   YmmMMMM   MMM     MMM      mM'            'YMUP YMMYMM      78   YUMMMMMP  MMM   MMP  ")
    print()
    print(Fore.BLUE + " Ensure your OPENAI_API_KEY                                                                d8rh8r - 2023  ")
    print(Fore.CYAN + " is set as an environmental variable                                   Dark assistance in bright colours  ")



def open_file(filepath):  # Open and read a file
    with open(filepath, 'r', encoding='UTF-8') as infile:
        return infile.read()

def save_file(filepath, content):  # Create a new file or overwrite an existing one.
    with open(filepath, 'w', encoding='UTF-8') as outfile:
        outfile.write(content)

def append_file(filepath, content):  # Create a new file or append an existing one.
    with open(filepath, 'a', encoding='UTF-8') as outfile:
        outfile.write(content)

openai.api_key = os.getenv("OPENAI_API_KEY")
## openai.api_key = "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"  # Grabs your OpenAI key from a file

def gpt_3(prompt):  # Sets up and runs the request to the OpenAI API
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.1,
            max_tokens=600,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        text = response['choices'][0]['text'].strip()
        return text
    except openai.error.APIError as e:
        print(Fore.RED + "\nError communicating with the API.")  # Color the error message red
        print(Fore.RED + f"\nError: {e}")  # Color the detailed error message red
        print(Fore.YELLOW + "\nRetrying...")  # Color the retry message yellow
        print(Style.RESET_ALL)  # Reset the color
        return gpt_3(prompt)
    
banner()
while True:
    request = input(Fore.GREEN + "\nEnter request: ")  # Color the input prompt green
    print(Style.RESET_ALL, end='')  # Reset the color
    if not request:
        break
    if request == "quit":
        break
    prompt = open_file("prompt4.txt").replace('{INPUT}', request)
    command = gpt_3(prompt)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True)
    print(Fore.CYAN + "\n" + command + "\n")  # Color the command cyan
    print(Style.RESET_ALL, end='')  # Reset the color
    with process:
        for line in process.stdout:
            print(Fore.MAGENTA + line, end='', flush=True)  # Color the output magenta
    print(Style.RESET_ALL, end='')  # Reset the color
    exit_code = process.wait()
    append_file("command-log.txt", "Request: " + request + "\nCommand: " + command + "\n\n")
