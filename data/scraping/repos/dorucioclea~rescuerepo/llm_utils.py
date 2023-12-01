#!/usr/bin/env python
import os
import re

from loguru import logger
from typing import List, Optional, Union

import openai

from langchain.chat_models import ChatAnthropic
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


def authenticate_openai(openai_api_key: str) -> None:
    openai.api_key = openai_api_key


def list_openai_engines() -> Optional[List[dict]]:
    # Ensure the user is authenticated
    if not openai.api_key:
        raise ValueError(
            "OpenAI API key not set. Please authenticate using 'authenticate_openai(api_key)' function."
        )

    try:
        response = openai.Engine.list()
        engines = response["data"]
        return engines
    except openai.error.APIError as api_error:
        logger.error(f"API error: {api_error}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None


def extract_code_blocks_from_llm_output(text: str) -> List[str]:
    code_blocks = re.findall(r"```([\s\S]*?)```", text)
    code_blocks = [code_block.strip() for code_block in code_blocks]
    return code_blocks


def get_code_block_openai(
    prompt: str, model: str = "gpt-3.5-turbo-0301" # Switch to "gpt-4-32k" if you can
) -> Union[str, None]:
    # Ensure the user is authenticated
    if not openai.api_key:
        raise ValueError(
            "OpenAI API key not set. Please authenticate using 'authenticate_openai(api_key)' function."
        )

    # Set up the API request
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "n": 1,
        "stop": None,
        "temperature": 1.0,
    }

    try:
        logger.info("Sending request to OpenAI API")
        logger.debug(f"Using the following parameters:\n  Model:\n{model}\n  Prompt:\n{prompt}")

        # Make the API request
        response = openai.ChatCompletion.create(**data)

        # Check if there are choices in the response
        if not response.choices:
            logger.error(
                "Error: The API response does not contain any generated text."
            )
            return None

        logger.trace(str(response))
        logger.trace(str(response.choices[0].message.content))
        generated_text = response.choices[0].message.content.strip()
        logger.success("Generated text successfully received.")

        code_blocks = extract_code_blocks_from_llm_output(generated_text)
        if len(code_blocks) > 0:
            return code_blocks[0]
        else:
            return generated_text

    except openai.error.APIError as api_error:
        logger.error(f"API error: {api_error}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None


def get_code_block_anthropic(
    prompt: str, model: str = "claude-v1.3-100k", #@param ["claude-v1.3","claude-v1.3-100k","claude-v1.2","claude-v1.0"]
) -> Union[str, None]:
    
    anthropic_chat = ChatAnthropic(
        model=model,
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"]
    )

    try:
        logger.info("Sending request to Anthropic Claude API")
        logger.debug(f"Using the following parameters:\n  Model:\n{model}\n  Prompt:\n{prompt}")

        # Make the API request
        messages = [
            HumanMessage(content=prompt)
        ]
        ai_response: AIMessage = anthropic_chat(messages)

        # Check if there are choices in the response
        if not ai_response.content:
            logger.error(
                "Error: The AIMessage response from Anthropic does not contain any generated text."
            )
            return None

        logger.trace(ai_response.content)
        logger.trace(ai_response.additional_kwargs)
        logger.trace(ai_response)
        generated_text = ai_response.content
        logger.success("Generated text successfully received.")

        code_blocks = extract_code_blocks_from_llm_output(generated_text)
        if len(code_blocks) > 0:
            return code_blocks[0]
        else:
            return generated_text
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None


if __name__ == "__main__":
    logger.info(
        get_code_block_anthropic(
            prompt = "Generate a sample python code that adds two numbers. Generate only code, and put it between two markdown code block markers.",
            model = "claude-v1.3"
        )
    )
