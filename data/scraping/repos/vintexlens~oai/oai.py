#!/usr/bin/env python3
import argparse
import os
import platform
import sys
import json
import io

from rich import pager
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

from resources import config
# from resources.conduit import get_completion
from resources.conduit import get_chat, get_models


console = Console()
version = "0.4.0"
_session_file_=".messages.json"

def get_lindata():
    lindata = "Users Kernel:" + platform.platform() + "\n" \
              "Users OS:" + os.uname().version + "\n" \
              "Users Shell:" + os.environ.get("SHELL", "").split("/")[-1] + "\n\n"
    return lindata


def post_completion(openai_response):
    if config.get_expert_mode() != "true":
        openai_response += '\n\n[Notice] OpenAI\'s models have limited knowledge after 2020. Commands and versions' \
                           'may be outdated. Recommendations are not guaranteed to work and may be dangerous.' \
                           'To disable this notice, switch to expert mode with `oai --expert`.'
    return openai_response

def get_session():
    with open(_session_file_) as sf:
      messages=json.load(sf)
    return messages

def put_session(messages):
    try:
      to_unicode = unicode
    except NameError:
      to_unicode = str
    with open(_session_file_, 'w', encoding='utf8') as outfile:
      str_ = json.dumps(messages,
                       indent=4, sort_keys=True,
                       separators=(',', ': '), ensure_ascii=False)
      outfile.write(to_unicode(str_))

def main():
    desc = "This tool sends a query to OpenAIs Chat API from the command line.\n\n"\
           "A new chat session is started with -n <pre-info> and gives the opportunity to\n"\
           "provide pre-information to your question\n\n"\
           "Report any issues at: https://github.com/draupner1/oai/issues"
    epilog = "Please note that the responses from OpenAI's API are not guaranteed to be accurate and " \
            "use of the tool is at your own risk.\n"

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(prog='oai - CLI assistant',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description=desc,
                                     epilog=epilog)

      
    # Add arguments for expert mode, API key reset, version, and prompt
    parser.add_argument('-n', '--new', action="store_true", help='Start New Chat', dest='new')
    parser.add_argument('-l', '--linux', action="store_true", help='Include an assistent message with Kernel/OS/shell', dest='linux')
    parser.add_argument('-m', '--model', action="store_true", help='List models available via OpenAIs API', dest='model')
    parser.add_argument('-x', '--expert', action="store_true", help='Toggle warning', dest='expert')
    parser.add_argument('-i', '--key', action="store_true", help='Reset API key', dest='apikey')
    parser.add_argument('-v', '--version', action="store_true", help=f'Get Version (hint: it\'s {version})', dest='version')
    parser.add_argument('--licenses', action="store_true", help='Show oai & Third Party Licenses', dest='licenses')
    parser.add_argument('prompt', type=str, nargs='?', help='Prompt to send')
    args = parser.parse_args()

    if args.new:
        console.status("Starting a new chat session")
        if os.path.exists(_session_file_):
          os.remove(_session_file_)
        if args.prompt:
          prompt = args.prompt
        else:
          prompt = ""
        pprompt = f"{prompt}\n\n" \
                  f"Response Format: Markdown\n"
        messages=[{'role':'assistant', 'content':pprompt}]
        put_session(messages)
        sys.exit()
 
    if args.linux:
        prompt = get_lindata()
        if os.path.isfile(_session_file_):
          messages=get_session()
          messages.append({'role':'user', 'content':prompt})
        else:
          messages=[{'role':'user', 'content':prompt}]
        put_session(messages)  
        sys.exit()
 
    if args.model:
        model_list = get_models()
        for mod in model_list:
          print(mod['id'])
        sys.exit()

    if args.version:
        console.print("oai version: " + version)
        sys.exit()
 
    if args.licenses:
        # print LICENSE file with pagination
        with console.pager():
            try:
                with open("LICENSE", "r") as f:
                    console.print(f.read())
            except FileNotFoundError:
                with open("/app/bin/LICENSE", "r") as f:
                    console.print(f.read())
        sys.exit()

    config.check_config(console)
    if args.apikey:
        config.prompt_new_key()
        sys.exit()

    if args.expert:
        config.toggle_expert_mode()
        sys.exit()

    if not args.prompt:
        prompt = Prompt.ask("Documentation Request")
        if prompt == "":
            print("No prompt provided. Exiting...")
            sys.exit()
    else:
        prompt = args.prompt
    if os.path.isfile(_session_file_):
      messages = get_session()
      messages.append({'role':'user', 'content':prompt})
    else:
      messages=[{'role':'user', 'content':prompt}]

    with console.status(f"Phoning a friend...  ", spinner="pong"):
        openai_response = post_completion(get_chat(messages))
        console.print(Markdown(openai_response.strip()))
        messages.append({'role':'assistant', 'content':openai_response.strip()})
        put_session(messages)


if __name__ == "__main__":
    main()
