# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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
import os
import time
import openai
import argparse
import bittensor
from typing import List, Dict, Optional

from prompting.baseminer.miner import Miner
from prompting.protocol import Prompting


class OpenAIMiner(Miner):
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        """
        Adds OpenAI-specific arguments to the command line parser.

        This class method introduces command-line arguments that pertain specifically to the
        OpenAI GPT model's completion settings, such as temperature, max tokens, and model name.
        Developers extending or utilizing this method can easily customize the miner's operation
        by providing these arguments when starting the miner.

        Args:
            parser (argparse.ArgumentParser):
                The command line argument parser to which the OpenAI-specific arguments should be added.

        Note:
            Consider adding or adjusting arguments here if introducing new features or parameters
            related to OpenAI's model completion.
        """
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

    def config(self) -> "bittensor.Config":
        """
        Provides the configuration for the OpenAIMiner.

        This method returns a configuration object specific to the OpenAIMiner, containing settings
        and parameters related to the OpenAI model and its interaction parameters. The configuration
        ensures the miner's optimal operation with the OpenAI model and can be customized by adjusting
        the command-line arguments introduced in the `add_args` method.

        Returns:
            bittensor.Config:
                A configuration object specific to the OpenAIMiner, detailing the OpenAI model settings
                and operational parameters.

        Note:
            If introducing new settings or parameters for OpenAI or the miner's operation, ensure they
            are properly initialized and returned in this configuration method.
        """
        parser = argparse.ArgumentParser(description="OpenAI Miner Configs")
        self.add_args(parser)
        return bittensor.config(parser)

    def __init__(self, api_key: Optional[str] = None, *args, **kwargs):
        super(OpenAIMiner, self).__init__(*args, **kwargs)
        if api_key is None:
            raise ValueError(
                "OpenAI API key is None: the miner requires an `OPENAI_API_KEY` defined in the environment variables or as an direct argument into the constructor."
            )
        if self.config.wandb.on:
            self.wandb_run.tags = self.wandb_run.tags + ("openai_miner",)
        openai.api_key = api_key

    def prompt(self, synapse: Prompting) -> Prompting:
        """
        Overrides the Miner's abstract `prompt` method to process incoming requests using OpenAI.

        This method makes use of the OpenAI GPT model to generate completions for the incoming requests.
        When implementing or extending this method, developers should ensure that the `synapse` object
        contains both `roles` and `messages` fields. The `roles` field describes the type of each message
        (e.g., system, user), while the `messages` field contains the actual content of each message.

        Args:
            synapse (Prompting):
                The incoming request object. Must contain:
                    - `roles`: List of roles for each message, e.g., ["system", "user"].
                      Describes the origin or type of each message.
                    - `messages`: List of actual message content corresponding to each role.
                The combination of roles and messages forms a conversation context for the model.

        Returns:
            Prompting:
                The response object containing the model's generated completion. This is essentially
                the filled synapse request object with an added `completion` field which contains the
                model's response.

        Note:
            Developers extending this method should ensure proper handling of both `roles` and `messages`
            from the `synapse` object to maintain the conversation context. Additionally, consider adjusting
            OpenAI-specific parameters (e.g., temperature, max_tokens) in the config to tailor the response
            generation process.
        """
        messages = [
            {"role": role, "content": message}
            for role, message in zip(synapse.roles, synapse.messages)
        ]
        bittensor.logging.debug(f"messages: {messages}")
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
        synapse.completion = resp
        bittensor.logging.debug(f"completion: {resp}")
        return synapse


if __name__ == "__main__":
    """
    Main execution point for the OpenAIMiner.

    This script initializes and runs the OpenAIMiner, which connects to the Bittensor network
    and uses the OpenAI model for processing incoming requests. The miner continuously listens
    for these requests, generating responses using the OpenAI GPT model's completion capabilities.

    Before running, ensure that the `OPENAI_API_KEY` environment variable is set with a valid
    OpenAI API key to authorize the model's completions.

    Note:
        When executing the script, the miner runs indefinitely, periodically logging its status.
        To stop the miner, use a keyboard interrupt or ensure proper termination of the script.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")

    with OpenAIMiner(api_key=openai_api_key):
        while True:
            print("running...", time.time())
            time.sleep(1)
