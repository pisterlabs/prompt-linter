"""
Title: Asynchronous OpenAI API Wrapper with Error Handling

Description: A Python script to interact with the OpenAI API asynchronously, with built-in error handling and retry logic. This script is designed to handle multiple prompts with the ability to define custom functions for more complex queries.
Created from guidance at: https://platform.openai.com/docs/guides/gpt/function-calling

Updated: rev 1 - 16/11/2023 for new end point and gpt-4-1106-preview [openai==1.2.4]

Author: @CivilEngineerUK
"""

import asyncio
import json
import openai

class OpenAI_ASync:
    def __init__(self):
        self.client = openai.AsyncOpenAI()

    async def openai_api_call(self, func, **kwargs):
        max_retries = 5
        retry_count = 0
        base_wait_time = 1

        while retry_count <= max_retries:
            try:
                return await func(**kwargs)
            except (openai.error.APIError,
                    openai.error.APIConnectionError,
                    openai.error.RateLimitError,
                    openai.error.Timeout,
                    openai.error.ServiceUnavailableError) as e:
                error_messages = {
                    openai.error.APIError: "Issue on OpenAI's side.",
                    openai.error.APIConnectionError: "Failed to connect to OpenAI API.",
                    openai.error.RateLimitError: "Rate limit exceeded.",
                    openai.error.Timeout: "Request timed out.",
                    openai.error.ServiceUnavailableError: "Service unavailable."
                }
                error_message = error_messages.get(type(e), "Unknown error.")
                print(f"{error_message} Retrying... {2 ** retry_count}s")
                retry_count += 1
                await asyncio.sleep(base_wait_time * (2 ** retry_count))
            except (openai.error.AuthenticationError, openai.error.InvalidRequestError) as e:
                error_messages = {
                    openai.error.AuthenticationError: "Authentication error. Check API key or token.",
                    openai.error.InvalidRequestError: "Invalid request. Check parameters and retry."
                }
                print(error_messages.get(type(e), "Unknown error."))
                return None
            except Exception as e:
                print(f"Unexpected error: {e}. Exiting...")
                return None

        print("Max retries reached. Exiting...")
        return None

    async def handle_multiple_prompts(self, prompts, model, functions=None,
                                      function_call="auto", temperature=0.0):
        tasks = [self.create_chat_completion(prompt, model, functions, function_call,
                                             temperature=temperature) for prompt in prompts]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        return responses

    async def create_chat_completion(self, prompt, model="gpt-3.5-turbo-0613", functions=None, function_call="auto",
                                     temperature=0.0):
        async def api_call():
            return await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                functions=functions,
                function_call=function_call,
                temperature=temperature
            )



        response = await self.openai_api_call(api_call)

        return response


# Usage example:

# Define functions for function calling
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]


# Create an instance of the OpenAI_ASync class
openai_async = OpenAI_ASync()

# Define the prompts
prompts = [
    "What's the weather like in Boston?",
    "Weather in Seattle?",
    "Is it sunny in London?"
]

# Define model
model = "gpt-4-1106-preview"

async def main() -> None:

    # Get the responses
    responses = await openai_async.handle_multiple_prompts(
        prompts, model, functions, function_call="auto", temperature=0.0)

    for response in responses:
        # Extracting the arguments into dict
        arguments_dict = json.loads(response.choices[0].message.function_call.arguments)

        print(arguments_dict)

asyncio.run(main())