import openai
OPENAI_VER_MAJ = int(openai.__version__.split(".")[0])
if OPENAI_VER_MAJ >= 1:
    from openai import APIError, AuthenticationError, APIConnectionError
    from pydantic import BaseModel as CompletionObject
else:
    from openai.error import APIError, AuthenticationError, APIConnectionError
    from openai.openai_object import OpenAIObject as CompletionObject

def run_chat_completion(
    model_name,
    messages,
    token,
    endpoint,
    max_tokens=300,
    n=1,
    stream=False,
    stop=None,
    temperature=0.0,
    top_p=1.0,
    frequency_penalty=0,
    presence_penalty=0
):
    openai.api_key = token
    if OPENAI_VER_MAJ > 0:
        openai.base_url = endpoint + "/v1"
        client = openai.OpenAI(api_key=token)
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            stream=stream,
            n=n,
            stop=stop,
            top_p=top_p,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
    else:
        openai.api_base = endpoint + "/v1"
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            stream=stream,
            n=n,
            stop=stop,
            top_p=top_p,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

    if OPENAI_VER_MAJ >= 1:
        return completion.model_dump(exclude_unset=True)
    else:
        return completion
    