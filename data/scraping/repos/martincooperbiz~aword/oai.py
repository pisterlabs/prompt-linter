# -*- coding: utf-8 -*-

import json
import logging
from pprint import pformat
from typing import List, Any, Dict

import openai
import tiktoken

from tenacity import retry, wait_random_exponential
from tenacity import stop_after_attempt, retry_if_not_exception_type

import aword.errors as E


GPT_MODEL = "gpt-3.5-turbo-0613"
Api_loaded = False


def ensure_api(api_key):
    global Api_loaded
    if not Api_loaded:
        if not api_key:
            raise E.AwordError('Cannot find an openai key in the environment, try setting '
                               'the AWORD_OPENAI_KEY environment variable')
        openai.api_key = api_key
        Api_loaded = True


@retry(wait=wait_random_exponential(min=1, max=20),
       stop=stop_after_attempt(6),
       retry=retry_if_not_exception_type(openai.InvalidRequestError))
def fetch_embeddings(text_or_tokens_array: List[str],
                     model_name: str) -> List[List[float]]:
    return [r['embedding'] for r in
            openai.Embedding.create(input=text_or_tokens_array,
                                    model=model_name)["data"]]


def get_embeddings(chunked_texts: List[str],
                   model_name: str) -> List[float]:
    # Split text_chunks into shorter arrays of max length 100
    max_batch_size = 100
    text_chunks_arrays = [chunked_texts[i:i+max_batch_size]
                          for i in range(0, len(chunked_texts), max_batch_size)]

    embeddings = []
    for text_chunks_array in text_chunks_arrays:
        embeddings += fetch_embeddings(text_chunks_array,
                                       model_name)

    return embeddings


@retry(wait=wait_random_exponential(min=1, max=20),
       stop=stop_after_attempt(6),
       retry=retry_if_not_exception_type(openai.InvalidRequestError))
def chat(model_name: str,
         system_prompt: str,
         user_prompt: str):
    return openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )["choices"][0]["message"]["content"]


@retry(wait=wait_random_exponential(min=1, max=20),
       stop=stop_after_attempt(6),
       retry=retry_if_not_exception_type(E.AwordError))
def chat_completion_request(messages: List[Dict],
                            functions: List[Dict] = None,
                            call_function: str = None,
                            temperature: float = 1,  # 0 to 2
                            model_name: str = GPT_MODEL,
                            attempts: int = 4) -> Dict:
    args = {'model': model_name,
            'messages': messages,
            'temperature': temperature,
            'n': 1}

    if functions:
        args['functions'] = functions

    if call_function:
        if not functions:
            raise E.AwordError(f'Cannot call function {call_function} if no functions are defined')

        found_function = len([fdesk for fdesk in functions if fdesk['name'] == call_function])
        if not found_function:
            raise E.AwordError(f'Cannot call undefined function {call_function}')
        if found_function > 1:
            raise E.AwordError(f'Found more than one definitions of {call_function}')

        args['function_call'] = {'name': call_function}

    try:
        logger = logging.getLogger(__name__)
        logger.info('Calling openai.ChatCompletion with model %s and temperature %.1f',
                    model_name,
                    temperature)
        logger.debug('openai.ChatCompletion arguments:\n\n%s', pformat(args))
        # Should check finish_reason in case it is 'length', which
        # would mean too many tokens.
        # https://platform.openai.com/docs/api-reference/chat/object#chat/object-finish_reason

        response = openai.ChatCompletion.create(**args)
        meta = {'model': response['model'],
                'prompt_tokens': response['usage']['prompt_tokens'],
                'completion_tokens': response['usage']['completion_tokens'],
                'total_tokens': response['usage']['total_tokens']}

        # The message can have either call_function or a content.
        message = response["choices"][0]["message"]
        function_call = message.get('function_call', None)
        if function_call:
            try:
                return {'call_function': function_call['name'],
                        'with_arguments': json.loads(function_call['arguments']),
                        'success': True,
                        **meta}
            except:
                if attempts:
                    return chat_completion_request(messages=messages,
                                                   functions=functions,
                                                   call_function=call_function,
                                                   temperature=temperature/2,
                                                   model_name=model_name,
                                                   attempts=attempts-1)
                return {'success': False,
                        **meta}
        return {'reply': message['content'],
                'success': True,
                **meta}
    except openai.InvalidRequestError as exc:
        raise E.AwordModelRequestError('Invalid request error from OpenAI') from exc
    # TODO: if it is a RateLimitError we should check if the rate
    # limit comes from tokens. If it does, we should not retry (raise
    # an AwordError) because the recovery time with gpt-4 is 6
    # min. Actually, we should only try again if it is a
    # RateeLimitError where the limiting factor is messages per
    # minute.


def get_tokenizer(encoding) -> Any:
    return tiktoken.get_encoding(encoding)
