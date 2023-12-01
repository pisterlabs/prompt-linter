import getpass
import json
import os
import os.path
from copy import copy
from dataclasses import dataclass
from typing import Union, List

import openai


# todo: add the ability to override specific config values
@dataclass
class QueryConfig:
    model: str = "text-davinci-003"
    max_tokens: int = 512
    temperature: float = 0.9
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Union[str, List[str]] = None
    user: str = getpass.getuser()

    def update(self, **kwargs):
        self.__dict__.update(kwargs)


DEFAULT_QUERY_CONFIG = QueryConfig()


class OpenaiWrapper:
    api = openai

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api.api_key = self.api_key

    def _query(self, prompt, config):
        # make sure api_key is correct
        self.api.api_key = self.api_key

        response = self.api.Completion.create(
            model=config.model,
            prompt=prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            n=config.n,
            stream=config.stream,
            stop=config.stop,
            user=config.user,
        )

        return response

    def query(self, prompt: str, config: QueryConfig = None, **kwargs) -> str:
        if config is None:
            config = DEFAULT_QUERY_CONFIG
        config = copy(config)
        config.update(**kwargs)

        # todo: check prompt length and response length

        response = self._query(prompt, config)
        return response.choices[0].text

    # make a query to openai_wrapper that uses the cheaper - curie - model
    def query_cheap(self, prompt: str, **kwargs) -> str:
        if "model" in kwargs:
            raise ValueError("For cheap query model cannot be overridden")
        return self.query(prompt, model="text-curie-001", **kwargs)

    def edit(self, prompt, instruction, config: QueryConfig = DEFAULT_QUERY_CONFIG, **kwargs):
        if config is None:
            config = DEFAULT_QUERY_CONFIG
        config = copy(config)
        config.update(**kwargs)
        response = self.api.Edit.create(
            model="text-davinci-edit-001",
            input=prompt,
            instruction=instruction,
            temperature=config.temperature,
            n=config.n,
            top_p=config.top_p,
        )
        return response.data[0].text


registry = {}

possible_filenames = [
    '.openai_api_key',
    'secrets.json',
    'secrets.txt',
]

possible_key_locations = [
    os.path.expanduser('~'),
    os.path.abspath(os.path.dirname(__file__)),
    os.path.abspath('')
]


def validate_api_key(api_key: str):
    old_key = openai.api_key
    try:
        openai.api_key = api_key
        openai.Engine.list()
        openai.api_key = old_key
        return True
    except:
        openai.api_key = old_key
        return False


def get_openai_wrapper(api_key=None) -> OpenaiWrapper:
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    # if the object already exists, return it
    if api_key in registry:
        return registry[api_key]
    if api_key is None:
        for location in possible_key_locations:
            if api_key is not None:
                break
            for filename in possible_filenames:
                full_path = os.path.join(location, filename)
                if os.path.exists(full_path):
                    with open(full_path, 'r') as f:
                        data = f.read()
                        if filename.endswith('.json'):
                            data = json.loads(data)
                            if 'openai_api_key' in data:
                                api_key = data['openai_api_key']
                                break
                        elif filename.endswith('.txt'):
                            # noinspection PyTypeChecker
                            d = dict([line.split(':', 1) for line in data.splitlines()])
                            if 'openai_api_key' in d:
                                api_key = d['openai_api_key']
                                break
                        elif filename.startswith('.'):
                            # check it's only one line:
                            if len(data.strip().splitlines()) == 1:
                                api_key = data.strip()
                                break
        if api_key is None:
            while not validate_api_key(api_key):
                message = "Please enter your OpenAI API key. You can find it at https://beta.openai.com/account/api-keys: "
                api_key = getpass.getpass(message)
                if not validate_api_key(api_key):
                    print("Invalid API key. Please try again.")
            # save token to file, with user confirmation
            print("Key received")
            res = input("Save token to file? [y/n]")
            if res.lower() == 'y':
                default_path = os.path.join(os.path.expanduser('~'), '.openai_api_key')
                path = input(f"Enter path to save token to [{default_path}]: ")
                if path == '':
                    path = default_path
                with open(path, 'w') as f:
                    f.write(api_key)

    openai_wrapper = OpenaiWrapper(api_key)
    registry[api_key] = openai_wrapper
    return openai_wrapper
