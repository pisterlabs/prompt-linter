from typing import Any, Dict

import openai
from hugchat import hugchat

from .utils import extract_code


def call_api(
    api: str,
    prompt: str,
    library: str,
    params: Dict[str, Any] = {},
    api_key: str = None,
    **kwargs,
) -> str:
    prompt = f"""
Reply with a python module using {library}; \
this module will have one function with arguments {', '.join(kwargs.keys())}; \
this function will perform what is described by the following instructions delimited by <<< and >>>; \
<<<{prompt}>>>;
verify that the reply has all the necessary imports, that it contains only valid python code, \
and that the keywords used are present in the official documentation of the libraries from which they came;
don't include any explanations in your reply, returning only python code.
    """

    print(prompt)

    if api == "openai":
        openai.api_key = api_key

        messages = [{"role": "user", "content": prompt}]
        reply = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
            **params,
        )
        return extract_code(reply.choices[0].message["content"])
    elif api == "hugchat":
        chatbot = hugchat.ChatBot()
        return extract_code(
            chatbot.chat(
                text=prompt,
                temperature=1e-6,
                **params,
            )
        )
