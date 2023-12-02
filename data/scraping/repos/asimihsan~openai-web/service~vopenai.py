from dataclasses import dataclass

import openai
import os
from dotenv import load_dotenv

load_dotenv()


class OpenAIWrapper:
    client: openai.AsyncOpenAI

    def __init__(self):
        if "OPENAI_API_KEY" not in os.environ:
            raise Exception("OPENAI_API_KEY not found in environment variables")
        self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def complete(
        self,
        conversation: list[dict],
        max_tokens=4095,
        temperature=0.0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        """
        Complete a conversation.

        :param conversation: A list of dictionary with two keys, "role" and "content".
        "role" is either "user" or "assistant" and "text" is the text of the message.
        :param max_tokens: The maximum number of tokens to generate.
        :param temperature: What sampling temperature to use. Higher values means the model will take more risks.
        :param top_p: An alternative to sampling with temperature, called nucleus sampling, where the model considers
        the results of the tokens with top_p probability mass.
        :param frequency_penalty: What sampling temperature to use. Higher values means the model will take more risks.
        :param presence_penalty: What sampling temperature to use. Higher values means the model will take more risks.
        :return: A generator that yields the text of the assistant's response.
        """

        async for chunk in await self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=conversation,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=True,
        ):
            choice = chunk.choices[0]
            finish_reason = choice.finish_reason  # None, "stop", or "length"
            delta = choice.delta

            if delta.role == "assistant":
                continue

            if finish_reason is None:
                yield delta.content
