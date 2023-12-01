#!/usr/local/opt/python/bin/python
import argparse
import os
import readline
import signal
import subprocess
import sys

import colorama
from colorama import Fore, Style
import openai
import requests
import debug

try:
    import gnureadline as readline
except ImportError:
    print('import gnu readline error')
    import readline

colorama.init()

EXAMPLES_CONTEXT = "Linux bash command to accomplish the task, you should return command that can be executed directly."

REVERSE_EXAMPLES_CONTEXT = "English description of Linux bash command"

EXAMPLES = [
    ["Find files ending in \"log\" in the root directory",
        "find / -name \"*.log\""],
    # ["Look up the IP address 12.34.56.78",        "nslookup 12.34.56.78"],
    # ["Create a git branch called feature1",        "git checkout -b feature1"],
    ["列举当前文件夹所有文件", "ls -a"]
]

MODEL = "gpt-3.5-turbo"

# openai.ChatCompletion.create = debug.mock_response


def get_command_openai(prompt):
    template_mess = [{'role': 'system', 'content': EXAMPLES_CONTEXT}]
    for exp in EXAMPLES:
        qus = {'role': 'user', 'content': exp[0]}
        ans = {'role': 'assistant', 'content': exp[1]}
        template_mess.append(qus)
        template_mess.append(ans)

    usr_promt = {'role': 'user', 'content': prompt}

    template_mess.append(usr_promt)

    results = openai.ChatCompletion.create(
        model=MODEL,
        messages=template_mess,
        max_tokens=200,
        temperature=0,
        stop=[ "<|endoftext|>"],
    )
    
    if results:
        return results['choices'][0]['message']['content']


def get_description_openai(command):
    template_mess = [{'role': 'system', 'content': REVERSE_EXAMPLES_CONTEXT}]
    for exp in EXAMPLES:
        qus = {'role': 'user', 'content': exp[1]}
        ans = {'role': 'assistant', 'content': exp[0]}
        template_mess.append(qus)
        template_mess.append(ans)

    usr_promt = {'role': 'user', 'content': command}

    template_mess.append(usr_promt)

    results = openai.ChatCompletion.create(
        model=MODEL,
        messages=template_mess,
        max_tokens=200,
        temperature=0,
        stop=["\n", "<|endoftext|>"],
    )
    if results:
        return results['choices'][0]['message']['content']


get_command = get_command_openai
get_description = get_description_openai


def reverse_pairs(ls):
    return [(b, a) for a, b in ls]

def readline_with_edit(prompt, hint = 'gpt powerd'):
    try:
        sys.stdout.write("\033[1A")
        sys.stdout.flush()
        
        sys.stdout.write(f"{hint}\n")
        sys.stdout.flush()
        
        readline.set_startup_hook(lambda: readline.insert_text(prompt))
        
        readline.add_history(prompt)
        
        readline.parse_and_bind("set enable-keypad on")
        readline.parse_and_bind("set editing-mode vi")  # 可选，使用 vi 编辑模式
        
        line = input(f"\001{Fore.GREEN}{Style.BRIGHT}\002<~ \001{Fore.CYAN}{Style.NORMAL}\002")
        return line.strip()
    except (EOFError, KeyboardInterrupt):
        return None
    finally:
        readline.set_startup_hook()

def main():
    CURRENT_JOB = None
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    while True:
        try:
            request = input(
                f'\001{Fore.GREEN}{Style.BRIGHT}\002~> \002')
        except EOFError:
            print("")
            print(
                f"{Fore.GREEN}{Style.BRIGHT}<~ {Fore.CYAN}{Style.NORMAL}Farewell, human.{Style.RESET_ALL}")
            sys.exit(0)
        except KeyboardInterrupt:
            print("")
            continue

        if not request.strip():
            continue

        if not request.startswith("Q:"):
            res = subprocess.run([request,], shell=True)
            continue

        print(f"🤖 {Style.BRIGHT}Thinking...")

        new_command = get_command(request)

        if not new_command:
            print("<~ Unable to figure out how to do that")
            continue
        try:
            edit_command = readline_with_edit(new_command)

        except (EOFError, KeyboardInterrupt):
            print(
                f"\n{Fore.RED}{Style.BRIGHT}<~ Canceled{Style.RESET_ALL}")
            continue

        CURRENT_JOB = subprocess.Popen(["bash", "-c", edit_command])
        try:
            CURRENT_JOB.wait()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
