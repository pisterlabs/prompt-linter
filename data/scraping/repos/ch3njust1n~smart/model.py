"""
User-defined functions that generate code using LLM.
"""

import os
import openai
import cohere
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# import google.generativeai as palm
from dotenv import load_dotenv

from generative.metaclasses import AbstractGenerativeModel

load_dotenv()


class GPT4(AbstractGenerativeModel):
    """
    OpenAI Chat Completion API wrapper

    Args:
        prompt (string): The source code as context for function to replace.

    Returns:
        Source code of the generated function.
    """

    @classmethod
    def generate(cls, prompt: str) -> str:
        openai.api_key = os.getenv("OPENAI_API_KEY")

        if openai.api_key is None:
            raise ValueError(
                "The OPENAI_API_KEY environment variable is not set. Please provide your OpenAI API key."
            )

        messages = [
            {"role": "system", "content": "You are an elite Python programmer."},
            {"role": "user", "content": prompt},
        ]

        llm_code = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL_GPT4"),
            messages=messages,
            temperature=float(os.getenv("TEMPERATURE", 0.7)),
        )

        return llm_code.choices[0].message.content


class GPT3(AbstractGenerativeModel):
    """
    OpenAI Completion API wrapper

    Args:
        prompt (string): The source code as context for function to replace.

    Returns:
        Source code of the generated function.
    """

    @classmethod
    def generate(cls, prompt: str) -> str:
        openai.api_key = os.getenv("OPENAI_API_KEY")

        if openai.api_key is None:
            raise ValueError(
                "The OPENAI_API_KEY environment variable is not set. Please provide your OpenAI API key."
            )

        llm_code = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL_GPT3"),
            prompt=prompt,
            temperature=float(os.getenv("TEMPERATURE", 0.7)),
            max_tokens=int(os.getenv("MAX_TOKENS", 3600)),
        )

        return llm_code.choices[0].text


class Claude(AbstractGenerativeModel):
    """
    Claude API wrapper

    Args:
        prompt (string): The source code as context for function to replace.

    Returns:
        Source code of the generated function.
    """

    @classmethod
    def generate(cls, prompt: str) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if api_key is None:
            raise ValueError(
                "The ANTHROPIC_API_KEY environment variable is not set. Please provide your ANTHROPIC API key."
            )

        model = Anthropic()

        llm_code = model.completions.create(
            prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
            stop_sequences=[HUMAN_PROMPT],
            model=os.getenv("ANTHROPIC_MODEL"),
            temperature=float(os.getenv("TEMPERATURE", 0.7)),
            max_tokens_to_sample=int(os.getenv("MAX_TOKENS", 3600)),
        )

        return llm_code.completion


class Cohere(AbstractGenerativeModel):
    """
    Cohere API wrapper

    Args:
        prompt (string): The source code as context for function to replace.

    Returns:
        Source code of the generated function.
    """

    @classmethod
    def generate(cls, prompt: str) -> str:
        api_key = os.getenv("COHERE_API_KEY")

        if api_key is None:
            raise ValueError(
                "The COHERE_API_KEY environment variable is not set. Please provide your COHERE API key."
            )

        co = cohere.Client(api_key)

        response = co.generate(
            model=os.getenv("COHERE_MODEL"),
            prompt=prompt,
            max_tokens=os.getenv("MAX_TOKENS"),
            temperature=os.getenv("TEMPERATURE"),
        )

        return response.generations[0].text


# class Palm(AbstractGenerativeModel):
#     """
#     PaLM API wrapper

#     Args:
#         prompt (string): The source code as context for function to replace.

#     Returns:
#         Source code of the generated function.
#     """

#     @classmethod
#     def generate(cls, prompt: str) -> str:
#         api_key = os.getenv("GOOGLE_API_KEY")
#         palm.configure(api_key=api_key)

#         if api_key is None:
#             raise ValueError(
#                 "The GOOGLE_API_KEY environment variable is not set. Please provide your GOOGLE API key."
#             )

#         llm_code = palm.generate_text(
#             model=os.getenv("GOOGLE_MODEL"),
#             prompt=prompt,
#             temperature=float(os.getenv("TEMPERATURE", 0.7)),
#             # The maximum length of the response
#             max_output_tokens=int(os.getenv("MAX_TOKENS", 1000)),
#         )

#         return llm_code.result
