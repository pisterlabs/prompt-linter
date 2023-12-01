import sys
import os
sys.path.join(os.path.dirname(os.path.dirname('/')))

import anthropic
import time

from dotenv import load_dotenv
from claude_options import SelectOptions
from claude_models import SelectModel

api_key = os.getenv('ANTHROPIC_API_KEY')

load_dotenv()

class AnthropicAPI():
    """
    A class that provides an interface to the Anthropic API for text generation.
    """
    def __init__(self, model=None) -> None:
        global api_key
        """
        Initializes a new instance of the AnthropicAPI class.

        Args:
            model (str): The name of the model to use for text generation.
        """
        self.client = anthropic.Client.acompletion_stream(api_key=api_key)
        self.model= model

    async def generate_text(self, prompt, params):
        """
        Generates text using the Anthropic API.

        Args:
            prompt (str): The prompt to use for text generation.
            params (dict): A dictionary of parameters to pass to the model.

        Returns:
            str: The generated text.
        """
        reply = None
        num_retries = 5
        options = SelectOptions(prompt=prompt, model=self.model,params=params)
        for attempt in range(num_retries):
            backoff = 2 ** (attempt + 2)
            try:
                response = await self.client.acomplete(
                    options
                )
                reply = response
                break

            except anthropic.HTTPError:
                if response.status_code != 502:
                    raise

                print(f"\n\nError: Bad gateway, retrying in {backoff} seconds...")
                time.sleep(backoff)
        if reply is None:
            raise RuntimeError("\n\nError: Failed to get Anthropic Response")

        return reply