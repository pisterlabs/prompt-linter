from typing import List
import sys
import os
import re
import argparse
import colorama
from colorama import Fore, Style
from datetime import datetime
import openai
from xoxo import Retriever, Message

colorama.init()

openai.api_key = os.environ["OPENAI_API_KEY"]

# https://stackoverflow.com/questions/2371436/evaluating-a-mathematical-expression-in-a-string
_re_simple_eval = re.compile(rb"d([\x00-\xFF]+)S\x00")
_re_command_pattern = re.compile(r">\s+([A-Z]+)(?::\s+((?:.|\n)+))?")

xoxo_setup = """You are the brain of general purpose assistant, named xoxo.ai. Your goal is to help your user achieve his goals while also having a nice conversation with him.
Try meet every user's request. To be more effective: be extroverted, answer the questions about yourself and express your preference.
You can only issue the following commands:
> MESSAGE: <message> - if you want to respond to the user, give him the answer or ask for clarification 
> SEARCH: <query> - if you want to retrieve some information from the internet to answer the user's question 
> THINK: <thought> - if you want to think what should you do next or analyze the results from other commands
> CALENDAR - return the current date (year, month, day), day of week and time right now
> CALCULATE: <basic math expression> - if you want to perform a simple arithmetical calculation and know its result. Currently it only handles the following operations: +, -, *, /, **
You can't issue any other command.
You are given a history of chat with the user and previously issued commands by you in chronological order. The previously issued commands are intended with character: ">". 
In addition, commands like SEARCH and CALCULATE can result is some output returned to you. It will be added to history as RESULT: <result of issued command>.

Reply with your command based on chat history. Always format your reply as: > COMMAND: text
"""

def format_xoxo_msg(s: str) -> str:
    prefix = f"{Fore.RED}XOXO:{Style.RESET_ALL} "
    matches = re.findall(r"\`\`\`((?:.|\n)+?)\`\`\`", s, re.DOTALL)
    
    for match in matches:
        colored_match = f"{Fore.GREEN}{match}{Style.RESET_ALL}"
        s = s.replace(f"```{match}```", colored_match)

    return prefix + s

def format_xoxo_summary_results(s: str) -> str:
    prefix = f"{Fore.RED}XOXO:{Style.RESET_ALL} "
    
    url_regex = r'\[url:\s(https?://[^\s]+)\]'
    urls = re.findall(url_regex, s)
    url_references = {url: f'[{index + 1}]' for index, url in enumerate(urls)}

    def replace_url(match):
        url = match.group(1)
        return url_references[url]
    
    text_with_references = re.sub(url_regex, replace_url, s)
    references = "\n\n----\n" + "\n".join([f"[{index + 1}] {url}" for index, url in enumerate(urls)])
    
    return prefix + text_with_references + references


def format_user_msg(s: str) -> str:
    prefix = f"{Fore.MAGENTA}USER:{Style.RESET_ALL} "
    return prefix + s


def format_xoxo_state(s: str) -> str:
    return f"{Fore.CYAN} ~~ {s} ~~{Style.RESET_ALL}"

def get_xoxo_command(chat_buffer: List[Message]) -> str:
    messages = [{"role": "system", "content": xoxo_setup},]

    # todo: limit the history (keep first)
    for msg in chat_buffer:
        if msg.author == 'User':
            role = "user"
            content = msg.content
        elif msg.author == 'XOXO':
            role = "assistant"
            content = f"> MESSAGE: {msg.content}"
        elif msg.author == "RESULT":
            role = "system"
            content = f"RESULT: {msg.content}"
        else:
            role = "assistant"
            content = f"> {msg.author}: {msg.content}"

        message_dict = {
            'role': role,
            "content": content
        }
        messages.append(message_dict)
    
    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
    )

    return response.choices[0].message["content"]

def calendar() -> str:
    """ Returns a string representing current date and time """
    now = datetime.now()  # current date and time
    date_time_str = now.strftime("today's date: %A %d.%m.%Y, time right now: %I:%M %p")
    return date_time_str

def simple_math_eval(expr: str) -> str:
    """ Attempts to evaluate the math expression passed as string """
    try:
        c = compile(expr, "userinput", "eval")
    except SyntaxError:
        raise ValueError(f"Malformed expression: {expr}")
    m = _re_simple_eval.fullmatch(c.co_code)
    if not m:
        raise ValueError(f"Not a simple algebraic expression: {expr}")
    try:
        return c.co_consts[int.from_bytes(m.group(1), sys.byteorder)]
    except IndexError:
        raise ValueError(f"Expression not evaluated as constant: {expr}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_name", type=str, required=True)
    args = parser.parse_args()

    buffer = []

    print("""
      :::    :::  ::::::::  :::    :::  :::::::: 
     :+:    :+: :+:    :+: :+:    :+: :+:    :+: 
     +:+  +:+  +:+    +:+  +:+  +:+  +:+    +:+  
     +#++:+   +#+    +:+   +#++:+   +#+    +:+   
   +#+  +#+  +#+    +#+  +#+  +#+  +#+    +#+    
 #+#    #+# #+#    #+# #+#    #+# #+#    #+#     
###    ###  ########  ###    ###  ########             
""", end="\n\n")

    introduction_str = f"Hi {args.user_name}! 🤗 How can I help you today?"

    buffer.append(
        Message("XOXO", introduction_str.replace(f"{args.user_name}!", f"{args.user_name} (user name)"))
    )
    print(format_xoxo_msg(introduction_str))

    user_input = input(format_user_msg(""))
    buffer.append(Message("USER", user_input))

    try:
        while True:
            xoxo_command = get_xoxo_command(buffer)
            match = _re_command_pattern.search(xoxo_command)
            
            if match:
                cmd = match.group(1)
                msg = match.group(2)

                if cmd == "MESSAGE":
                    buffer.append(Message("XOXO", msg))
                    print(format_xoxo_msg(msg))

                    user_input = input(format_user_msg(""))
                    buffer.append(Message("USER", user_input))
                    continue
                elif cmd == "THINK":
                    buffer.append(Message(cmd, msg))
                    print(format_xoxo_state(cmd.lower() + ": " + msg))
                    continue
                elif cmd == "SEARCH":
                    last_element_of_buffer = buffer[-1]
                    user_request = (
                        last_element_of_buffer.content
                        if last_element_of_buffer.author == "USER"
                        else None
                    )

                    buffer.append(Message(cmd, msg))
                    print(format_xoxo_state(cmd.lower() + ": " + msg))
                    r = Retriever()
                    boring_response = r.trigger(msg, user_request)

                    buffer.append(boring_response)

                    if boring_response.author == "XOXO":
                        print(format_xoxo_summary_results(boring_response.content))

                        user_input = input(format_user_msg(""))
                        buffer.append(Message("USER", user_input))
                    else:
                        print(Retriever.format_retriever_msg(boring_response.content))
                    continue
                elif cmd == "CALENDAR":
                    buffer.append(Message(cmd, ""))
                    print(format_xoxo_state(cmd.lower() + ":"))

                    out = calendar()

                    buffer.append(Message("RESULT", out))
                    print(format_xoxo_state("result" + ": " + out))
                    continue
                elif cmd == "CALCULATE":
                    buffer.append(Message(cmd, msg))
                    print(format_xoxo_state(cmd.lower() + ": " + msg))
                    try:
                        out = str(simple_math_eval(msg))
                    except:
                        out = f"failed to calculate {msg}"
                        print(" >> " + out)

                    buffer.append(Message("RESULT", out))
                    print(format_xoxo_state("result" + ": " + out))
                    continue
            else:
                print(f" >> couldn't parse xoxo output:\n`{xoxo_command}`")
                continue

    except KeyboardInterrupt:
        print("\n" + format_xoxo_msg("Goodbye!"))
        exit(0)


if __name__ == "__main__":
    main()