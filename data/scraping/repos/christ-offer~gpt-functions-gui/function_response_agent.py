import openai 
from openai import OpenAIError
import logging


def function_response_agent(
        prompt,
        system_message,
        model = "gpt-3.5-turbo-16k-0613",
        temperature = 0.0,
        top_p = 1.0,
        frequency_penalty = 0.0,
        presence_penalty = 0.0,
        stream: bool = False,
        function_response: any = None,
        function_name: str = None,
        message: str = None
        ):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=stream,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
                message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                },
            ],
        )
    except OpenAIError as error:
        logging.error(f"OpenAI API call failed: {str(error)}")
        return "OpenAI API call failed due to an internal server error." + function_response
    except openai.error.APIConnectionError as e:
        print(f"Failed to connect to OpenAI API: {e}")
        return "Failed to connect to OpenAI." + function_response
    except openai.error.RateLimitError as e:
        print(f"OpenAI API request exceeded rate limit: {e}")
        return "Requests exceed OpenAI rate limit." + function_response

    return response["choices"][0]["message"]