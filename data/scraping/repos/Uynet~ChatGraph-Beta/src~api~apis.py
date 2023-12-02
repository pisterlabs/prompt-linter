import os

import openai

from utils.util import Console

tokenCount = 0
def validateAPIKey(key):
    try:
        openai.api_key = key
        response = openai.Engine.list()
        return response, None
    except Exception as e:
        return None, e

def getTokenCount():
    return tokenCount

def chatGPT(messages,stream=False):
    apikey = os.environ.get("OPENAI_API_KEY")
    openai.api_key = apikey 

    # validate
    try:
        validateAPIKey(apikey)
    except Exception as e:
        Console.log("on error",e)
        raise

    # todo network errorの処理
    response= openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=stream,
    )
    tokens = response["usage"]["total_tokens"]
    global tokenCount
    tokenCount += tokens

    answer = response.choices[0]["message"]["content"]
    return answer

def chatGPTReq(messages,stream=True):
    apikey = os.environ.get("OPENAI_API_KEY")
    openai.api_key = apikey 

    # validate
    try:
        validateAPIKey(apikey)
    except Exception as e:
        Console.error("on error",e)
        raise

    # todo network errorの処理
    response= openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=stream,
    )

    # 返答を受け取り、逐次yield
    response_text = "" 
    for chunk in response:
        if chunk:
            content = chunk['choices'][0]['delta'].get('content')
            if content:
                response_text += content
                yield content
    else:  #
        messages += [{'role': 'assistant', 'content': response_text}]