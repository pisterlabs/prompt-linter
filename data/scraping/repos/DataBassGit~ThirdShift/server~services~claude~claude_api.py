import os
import time
import anthropic

from dotenv import load_dotenv
from services.claude.claude_options import SelectOptions
from services.claude.claude_prompt_tempalte import PromptTemplate

load_dotenv()

class AnthropicAPI():
    """
    A class that provides an interface to the Anthropic API for text generation.
    """
    print('Anthropic API Client:')
    def __init__(self, model="claud-v1") -> None:
        """
        Initializes a new instance of the AnthropicAPI class.

        Args:
            model (str): The name of the model to use for text generation.
        """
        api_key=os.getenv('ANTHROPIC_API_KEY')
        api_url=os.getenv('ANTHROPIC_URL')
        print(f"- API Key: {api_key}")
        print("- Initializing client.")
        self.model= model
        print(f"- selected Model: {self.model}")
        self.client = anthropic.Client(
            api_key=api_key
        )
        self.prompt_template = PromptTemplate().prompt_tempalte

    async def generate_text(self, prompt: str = "", max_tokens_to_sample: int = 1000):
        print("- Generating response")
        reply = None
        num_retries = 1
        options = SelectOptions(
            prompt = self.prompt_template(prompt),
            max_tokens_to_sample=max_tokens_to_sample,
            model=self.model
        )
        for attempt in range(num_retries):
            backoff = 2 ** (attempt + 2)
            try:
                response = await self.client.acompletion_stream(kwargs=options)
                async for data in response:
                    print(f"-- Repsonse: {data}")
                    reply = response
                    break

            except Exception as e:
                print(f"Error: {e}")

                print(f"\n\nError: Bad gateway, retrying in {backoff} seconds...")
                time.sleep(backoff)
        if reply is None:
            raise RuntimeError("\n\nError: Failed to get Anthropic Response")

        return reply