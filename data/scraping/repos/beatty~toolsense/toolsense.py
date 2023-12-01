import argparse
from datetime import datetime
import functools
import logging
import openai
import json
import os
import sys
from dotenv import load_dotenv
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

_SYSTEM_MESSAGE = "You are toolsense, a helpful assistant that uses tools. The hints in parentheses are the answers from tools you invoked."


def llm_tool(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"executing {func.__name__}")
        result = func(*args, **kwargs)
        logging.info(f"finished executing {func.__name__}")
        return result

    return wrapper


@llm_tool
def get_current_date_time(parameters: dict) -> str:
    current_time = datetime.now()
    return f"""The current date and time is: {current_time.strftime("%Y-%m-%d %H:%M:%S")}"""


_TOOL_GET_DATE_TIME = {
    "name": get_current_date_time.__name__,
    "description": "gets the current date and time",
    "parameters": {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "if you want a timezone other than UTC",
            },
        },
    },
}


@llm_tool
def get_current_location(parameters: dict) -> str:
    return "37.774900, -122.419400"


_TOOL_GET_CURRENT_LOCATION = {
    "name": get_current_location.__name__,
    "description": "gets the user's current location",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


@llm_tool
def get_weather(parameters: dict) -> str:
    timestamp = parameters.get("timestamp")
    location = parameters.get("location")
    return f"The weather for location {location} at {timestamp} is: Sunny and 65F"


_TOOL_GET_WEATHER = {
    "name": get_weather.__name__,
    "description": "gets the weather",
    "parameters": {
        "type": "object",
        "properties": {
            "timestamp": {
                "type": "string",
                "description": "date/time in ISO format",
            },
            "location": {
                "type": "string",
                "description": "lat/long location in format: 37.774900, -122.419400",
            },
        },
    },
}

_TOOLS = [_TOOL_GET_DATE_TIME, _TOOL_GET_WEATHER, _TOOL_GET_CURRENT_LOCATION]

_FUNCTIONS = [get_current_date_time, get_weather, get_current_location]


def main(*, model: str, base_url: str, prompt: str, max_steps: int = 5):
    parsed_url = urlparse(base_url)
    fqdn = parsed_url.netloc
    if fqdn == "api.openai.com":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logging.error(
                "OPENAI_API_KEY environment variable must be set (or in .env file) when using the OpenAI API"
            )
            sys.exit(1)
    else:
        api_key = None

    functions = {f.__name__: f for f in _FUNCTIONS}
    tools = [{"type": "function", "function": t} for t in _TOOLS]

    system_message = [
        {"role": "system", "content": _SYSTEM_MESSAGE},
    ]
    question_message = [
        {
            "role": "user",
            "content": f"{prompt} Provide a detailed answer and explain your reasoning.",
        }
    ]

    llm_client = openai.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"], timeout=120, base_url=base_url
    )

    step = 0
    memory = []

    while step < max_steps:
        logging.info(f"step {step}")
        step += 1

        messages = system_message + question_message + memory
        logging.debug(json.dumps(messages, indent=2))

        api_params = {
            "model": model,
            "temperature": 0.5,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 100,
            "stream": False,
        }

        response = llm_client.chat.completions.create(**api_params)
        response_message = response.choices[0].message

        tool_calls = response_message.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                logging.info(f"calling function {function_name}")
                function = functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_return_value = function(function_args)
                memory.extend(
                    [
                        {
                            "role": "user",
                            "content": f"(hint: answer from the {function_name} tool you invoked: {function_return_value})",
                        }
                    ]
                )
        else:
            return response_message.content

    logging.debug("max steps reached with no answer")
    return None


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", help="OpenAI-compatible base URL", required=True)
    parser.add_argument("--model", help="model name", required=False)
    parser.add_argument(
        "--max-steps",
        help="maximum steps (or llm calls) before giving up",
        required=False,
        default=10,
        type=int,
    )
    parser.add_argument("prompt", help="The prompt to process")

    args = parser.parse_args()

    logging.info(f"base url: {args.base_url}")
    logging.info(f"model: {args.model}")
    logging.info(f"max steps: {args.max_steps}")
    logging.info(f"prompt: {args.prompt}")

    response = main(
        base_url=args.base_url,
        model=args.model,
        max_steps=args.max_steps,
        prompt=args.prompt,
    )
    if response:
        print(response)
    else:
        print(f"[No answer found in {args.max_steps} steps.]")
