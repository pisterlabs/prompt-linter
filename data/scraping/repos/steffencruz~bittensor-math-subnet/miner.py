# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Steffen Cruz

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import torch
import copy
import bittensor as bt
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

import openai
import sympy
import wolframalpha


class BaseMiner(ABC):
    @classmethod
    def config(cls) -> "bt.Config":
        return config(cls)

    @classmethod
    @abstractmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        ...

    @abstractmethod
    def forward(self, messages: List[Dict[str, str]]) -> str:
        ...

    # @abstractmethod
    # def backward(self, messages: List[Dict[str, str]], response: str, rewards: "torch.FloatTensor") -> str:
    #     ...

    def __init__(
        self,
        config: "bt.Config" = None
    ):

        # Instantiate and check configs.
        # Grab super config.
        super_config = copy.deepcopy(config or BaseMiner.config())

        # Grab child config
        self.config = self.config()

    def __str__(self):
        return f"{self.__class__.__name__}()"


class SimpleEvalMiner(BaseMiner):
    """Basic miner that evaluates the input using python built-in eval function. It's almost always wrong."""

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--fuzz", type=float, default=0.0, help="Amount of noise to add to the model output.")

    def config(cls) -> "bittensor.Config":
        parser = argparse.ArgumentParser(description="Simple Eval Miner Configs")
        cls.add_args(parser)
        return bt.config(parser)

    def __init__(self, config):
        super(SimpleEvalMiner, self).__init__(config)
        self.model = lambda x: eval(x)

        bt.logging.info(f"Running {self} with config {self.config}")

    def forward(self, query ) -> str:
        """
        Call the model with the given inputs and return the numerical output.
        """
        try:
            solution = str(self.model(query) + torch.randn(1).item() * self.config.fuzz)
        except:
            solution = ''

        bt.logging.info(f'Forwarded inputs: {query!r} to outputs: {solution}')

        return solution

    def __str__(self):
        return f"{self.__class__.__name__}(fuzz={self.config.fuzz})"



class OpenAIMiner(BaseMiner):
    """Miner that uses OpenAI's GPT models to generate a response."""

    system_prompt: str = 'You are an excellent math student. You are taking a test in your math class. Answer each question with a short numerical solution. The solution must be either numeric (e.g. float(solution) is valid) or a string that can be converted to a float (e.g. eval(solution) returns a float).'

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--openai.api_key", type=str, help="openai api key")
        parser.add_argument(
            "--openai.suffix",
            type=str,
            default=None,
            help="The suffix that comes after a completion of inserted text.",
        )
        parser.add_argument(
            "--openai.max_tokens",
            type=int,
            default=100,
            help="The maximum number of tokens to generate in the completion.",
        )
        parser.add_argument(
            "--openai.temperature",
            type=float,
            default=0.4,
            help="Sampling temperature to use, between 0 and 2.",
        )
        parser.add_argument(
            "--openai.top_p",
            type=float,
            default=1,
            help="Nucleus sampling parameter, top_p probability mass.",
        )
        parser.add_argument(
            "--openai.n",
            type=int,
            default=1,
            help="How many completions to generate for each prompt.",
        )
        parser.add_argument(
            "--openai.presence_penalty",
            type=float,
            default=0.1,
            help="Penalty for tokens based on their presence in the text so far.",
        )
        parser.add_argument(
            "--openai.frequency_penalty",
            type=float,
            default=0.1,
            help="Penalty for tokens based on their frequency in the text so far.",
        )
        parser.add_argument(
            "--openai.model_name",
            type=str,
            default="gpt-3.5-turbo",
            help="OpenAI model to use for completion.",
        )

    @classmethod
    def config(cls) -> "bittensor.Config":
        parser = argparse.ArgumentParser(description="OpenAI Miner Configs")
        cls.add_args(parser)
        return bt.config(parser)

    def __init__(self, config, api_key: Optional[str] = None):

        super(OpenAIMiner, self).__init__(config)

        if api_key is None and self.config.openai.api_key is None:
            raise ValueError(
                "the miner requires passing --openai.api_key as an argument of the config or to the constructor."
            )
        openai.api_key = api_key or self.config.openai.api_key

        bt.logging.info(f"Running {self} with config {self.config}")


    def forward(self, question: str) -> str:

        messages = [{'role':'system', 'content':self.system_prompt}, {'role':'user', 'content':question}]
        resp = openai.ChatCompletion.create(
            model=self.config.openai.model_name,
            messages=messages,
            temperature=self.config.openai.temperature,
            max_tokens=self.config.openai.max_tokens,
            top_p=self.config.openai.top_p,
            frequency_penalty=self.config.openai.frequency_penalty,
            presence_penalty=self.config.openai.presence_penalty,
            n=self.config.openai.n,
        )["choices"][0]["message"]["content"]
        bt.logging.info(f'Called OpenAI with Messages: {messages} and got response: {resp}')

        return resp



class SymPyMiner(BaseMiner):
    """Miner that uses SymPy to solve equations. Slightly smarter than SimpleEvalMiner."""
    """
    Links
    https://docs.sympy.org/latest/guides/solving/index.html
    https://docs.sympy.org/latest/modules/evalf.html
    https://www.sympy.org/scipy-2017-codegen-tutorial/
    https://www.sympy.org/scipy-2017-codegen-tutorial/notebooks/23-lambdify-Tc99m.html
    https://www.sympy.org/scipy-2017-codegen-tutorial/notebooks/22-lambdify.html
    """
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--dsolve", type=bool, default=False, help="Whether to use dsolve or solve")
        parser.add_argument("--evalf", type=bool, default=False, help="Whether to use evalf or not")
        parser.add_argument("--subs", type=bool, default=False, help="Whether to use subs or not")

    def config(cls) -> "bittensor.Config":
        parser = argparse.ArgumentParser(description="SymPy Miner Configs")
        cls.add_args(parser)
        return bt.config(parser)

    def __init__(self, config):
        super(SymPyMiner, self).__init__(config)
        import sympy
        self.sympy = sympy

        bt.logging.info(f"Running {self} with config {self.config}")

    def forward(self, query ) -> str:
        """
        Call the model with the given inputs and return the numerical output.
        """
        try:
            solution = str(self.sympy.solve(query, dict=True)[0])
        except:
            solution = ''

        bt.logging.info(f'Forwarded inputs: {query!r} to outputs: {solution}')

        return solution



class WolframAlphaMiner(BaseMiner):
    """Miner that uses Wolfram Alpha to solve equations."""

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        # 2LR63R-TVJ4422GY6
        parser.add_argument("--wolframalpha.app_id", type=str, help="wolframalpha app id", default='2LR63R-TVJ4422GY6')


    @classmethod
    def config(cls) -> "bittensor.Config":
        parser = argparse.ArgumentParser(description="OpenAI Miner Configs")
        cls.add_args(parser)
        return bt.config(parser)

    def __init__(self, config):
        super(WolframAlphaMiner, self).__init__(config)
        self.client = wolframalpha.Client(wolframalpha.getfixture('API_key'))


    def forward(self, query ) -> str:
        """
        Forward query to Wolfram Alpha API
        """
        response = client.query(question)
        try:
            for pod in response.pods:
                if pod.title == 'Result':
                    print(pod)
                    print(pod.subpod.plaintext)
                    result = pod.subpod.plaintext
        except:
            result = ''

        return result



ALLOWED_MODELS = (SimpleEvalMiner, SymPyMiner, OpenAIMiner, WolframAlphaMiner)

def get_model(config):

    for model in ALLOWED_MODELS:
        if config.model_type == model.__class__.__name__:
            return model(config)

    raise ValueError(f"Model {config.model_type} not found. Allowed models are {ALLOWED_MODELS}")

