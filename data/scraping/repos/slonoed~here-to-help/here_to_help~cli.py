import argparse
import guidance
import os
from iterfzf import iterfzf
from here_to_help.prompts_parser import parse_text
from here_to_help.web import WebServer
from here_to_help.processor import run
import uvicorn
import readline

default_model = 'gpt-4'

def get_model(name):
    match name:
        case "gpt-3.5-turbo":
            return guidance.llms.OpenAI("gpt-3.5-turbo")
        case "gpt-4":
            return guidance.llms.OpenAI("gpt-4")
    exit(f"Model not found: {name}")

def run_server(parsed_prompts):
    server = WebServer(parsed_prompts)
    uvicorn.run(server.app, host="0.0.0.0", port=8000, reload=False)

def main():
    parser = argparse.ArgumentParser(description='Here To Help CLI tool.')
    default_prompts_file_path = os.path.expanduser("~/.hth_prompts")

    parser.add_argument('-p', '--prompts-file',
                        type=str,
                        help=f'Path to the prompts file (default: {default_prompts_file_path})',
                        default=default_prompts_file_path)
    parser.add_argument('--web', action='store_true', help='Run web server')
    parser.add_argument('--filter',
                        type=str,
                        help='Filter prompts by title', default=None)
    parser.add_argument('-i', '--interactive',
                        action='store_true',
                        help='Run any prompt in interactive mode', default=False)

    args = parser.parse_args()

    if not os.path.exists(args.prompts_file):
        exit(f"Prompts file not found: {args.prompts_file}")

    f = open(args.prompts_file, 'r')
    content = f.read()
    f.close()

    parsed_prompts = parse_text(content)

    if args.web:
        run_server(parsed_prompts)
        return

    titles = [prompt['title'] for prompt in parsed_prompts]
    prompt = None

    if args.filter:
        prompt = next((prompt for prompt in parsed_prompts if prompt['title'] == args.filter), None)
        if not prompt:
            exit("cant find prompt by filter")
    else:
        selected_title = iterfzf(titles)
        if not selected_title:
            exit(f"No prompt selected")
        prompt = next((prompt for prompt in parsed_prompts if prompt['title'] == selected_title), None)

    name_values = {}
    for name in prompt['inputs']:
        user_value = input(f"Please enter a value for {name}: ")
        name_values[name] = user_value

    r = run(prompt, name_values, args.interactive)
    print(r)

if __name__ == '__main__':
    main()
