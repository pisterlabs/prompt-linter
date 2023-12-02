import logging
import openai
import math
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

"""
    recursive_summarization(text)
    text: string

    returns: string

    recursively summarizes the text until it is less than 4096 tokens
"""
def recursive_summarization(text):
    if count_tokens(text) < 3500:
        return text
    
    logging.debug(f"summarizing {text}")
    CHOP = math.ceil(3000*2.3)
    def summarize(text):
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "user", "content": f"summarize in a couple sentences: {text[:CHOP]}"},
        ],
        temperature=0.1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"]
        )
        return resp["choices"][0]["message"]["content"]
    return recursive_summarization(summarize(text) + text[CHOP:])
            

"""
    count_tokens(text)
    text: string

    returns: int

    gets the token count of the text from tiktoken
"""
def count_tokens(text) -> int:
    return len(encoding.encode(text))