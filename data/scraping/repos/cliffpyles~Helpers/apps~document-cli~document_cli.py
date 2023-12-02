#!/usr/bin/env python3

import json
import yaml
import click
import inquirer
import openai
import os
from pathlib import Path
from inquirer.errors import ValidationError

openai.api_key = os.getenv("OPENAI_API_KEY")


class RangeValidator(object):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, _, value):
        try:
            int_value = int(value)
            if self.min_value <= int_value <= self.max_value:
                return value
            else:
                raise ValidationError("", reason=f"Value must be between {self.min_value} and {self.max_value}")
        except ValueError:
            raise ValidationError("", reason="Please enter a valid number")


def read_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content


def load_config(file_path):
    with open(file_path, 'r') as config_file:
        if file_path.endswith('.json'):
            return json.load(config_file)
        elif file_path.endswith('.yml') or file_path.endswith('.yaml'):
            return yaml.safe_load(config_file)
        else:
            raise ValueError('Invalid file format. Use JSON or YAML.')


def display_prompts(prompts, arguments):
    questions = []

    for prompt in prompts:
        prompt_key = prompt.get('key')
        prompt_type = prompt['type']
        kwargs = prompt['kwargs']

        if prompt_key and arguments.get(prompt_key) is not None:
            continue

        if prompt_type == 'text':
            question = inquirer.Text(**kwargs)
        elif prompt_type == 'checkbox':
            question = inquirer.Checkbox(**kwargs)
        elif prompt_type == 'radio':
            question = inquirer.List(**kwargs)
        elif prompt_type == 'range':
            min_value = kwargs.pop('min', None)
            max_value = kwargs.pop('max', None)
            if min_value is not None and max_value is not None:
                kwargs['validate'] = RangeValidator(min_value, max_value)
            question = inquirer.Text(**kwargs)
        elif prompt_type == 'file':
            question = inquirer.Text(**kwargs)
        else:
            raise ValueError(f'Invalid prompt type: {prompt_type}')

        questions.append(question)

    user_responses = inquirer.prompt(questions)
    responses = {**arguments, **user_responses}

    # Read the contents of the file for 'file' prompt type
    for prompt in prompts:
        prompt_key = prompt.get('key')
        prompt_type = prompt['type']
        if prompt_type == 'file' and responses.get(prompt_key) is not None:
            file_path = responses[prompt_key]
            responses[f"{prompt_key}_content"] = read_file(file_path)

    return {k: v for k, v in responses.items() if v is not None}


def generate_options(prompts):
    options = []

    for prompt in prompts:
        prompt_key = prompt.get('key')
        prompt_type = prompt['type']

        if prompt_key:
            if prompt_type == 'radio':
                choices = prompt['kwargs']['choices']
                option = click.Option(param_decls=[f'--{prompt_key}'],
                                      type=click.Choice(choices, case_sensitive=False),
                                      help=f'Pass your {prompt_key} preference as an argument.')
            elif prompt_type == 'file':
                option = click.Option(param_decls=[f'--{prompt_key}'],
                                      type=click.Path(exists=True, dir_okay=False, resolve_path=True),
                                      help=f'Pass the file path for {prompt_key} as an argument.')
            else:
                option = click.Option(param_decls=[f'--{prompt_key}'],
                                      type=str,
                                      help=f'Pass your {prompt_key} as an argument.')
            options.append(option)

    return options


def chat_with_gpt(message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": message}]
    )
    return response.choices[0].message['content'].strip()


def main(**kwargs):
    file_path = kwargs.pop('file', None) or config_file_path
    config = load_config(str(file_path))
    responses = display_prompts(config["prompts"], kwargs)

    # Construct the command to reproduce the current context
    command = "document"
    for k, v in responses.items():
        for prompt in config["prompts"]:
            prompt_key = prompt.get('key')
            prompt_type = prompt['type']
            if k == prompt_key and prompt_type != 'file':
                command += f" --{k} \"{v}\""
            elif k == prompt_key and prompt_type == 'file':
                command += f" --{k} \"{v}\""

    # Initialize the messages list with the system message
    messages = [{"role": "system", "content": config["context"]}]

    # Add user responses as separate messages
    for k, v in responses.items():
        for prompt in config["prompts"]:
            prompt_key = prompt.get('key')
            prompt_type = prompt['type']
            if k == prompt_key and prompt_type == 'file':
                messages.append({"role": "user", "content": f"{k}_path: {v}"})
                messages.append({"role": "user", "content": f"{k}: {responses[f'{k}_content']}"})
            elif k == prompt_key:
                messages.append({"role": "user", "content": f"{k}: {v}"})

    for message in config["messages"]:
        messages.append({"role": "user", "content": message})

    # Send messages to ChatGPT and display the response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    chatgpt_response = response.choices[0].message['content'].strip()
    click.echo(chatgpt_response)
    click.echo("\nCommand:")
    click.echo(command)
    click.echo("\n")

script_dir = Path(__file__).resolve(strict=False).parent
config_file_path = script_dir / './prompts.yaml'
config = load_config(str(config_file_path))
options = generate_options(config["prompts"])
main = click.Command('main', callback=main, params=options)

if __name__ == '__main__':
    main()