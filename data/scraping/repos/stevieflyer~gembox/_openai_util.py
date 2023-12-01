import os
import openai

# try to get system environment variable: OPENAI_API_KEY
try:
    api_key = os.environ["OPENAI_API_KEY"]
    print(f"Found openai api key: {api_key}")
    openai.api_key = api_key
except KeyError:
    print(f"No OPENAI_API_KEY environment variable found.")
    api_key = None


def chat_complete(msg: str, sys_msg: str = "You are a helpful assistant", openai_api_key: str = None,
                  model="gpt-3.5-turbo", raw_json: bool = False):
    """
    
    :param msg: (str) the message to be sent to the chatbot
    :param sys_msg: (str) system message, helping you to customize the chatbot
    :param model: (str) the model to be used
    :param openai_api_key: (str) the openai api key
    :param raw_json: (bool) whether to return the raw json response
    :return: (str | dict) the response from the chatbot
    """
    if openai_api_key is not None:
        openai.api_key = openai_api_key
    else:
        if openai.api_key is None:
            raise ValueError(
                "No OPENAI_API_KEY environment variable found. Please pass in the openai_api_key parameter or set the OPENAI_API_KEY environment variable.")

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": msg},
        ]
    )

    if raw_json:
        return response
    else:
        return response.choices[0].message.content


__all__ = ["chat_complete"]
