"""Deeper 2022, All Rights Reserved

TODO: Need to not be depended on OpenAI API, Too expensive
"""
import os
import logging
import json
from json.decoder import JSONDecodeError
from typing import Dict

import openai


_ExtractContext: Dict = {
    "5_categories": [],
    "5_keywords": [],
    "positiviety_scale": 100,
    "meaning": "",
    "follow_up_questions": [],
    # "openness": 0,
    # "conscientiousness": 0,
    # "extroversion": 0,
    # "agreeableness": 0, 
    # "neuroticism": 0
}

openai.api_key = os.getenv('OPENAI_APIKEY')

_OPENAI_PROMPT = f"Analyse the following tweet and turn it into a JSON object as follows: \n{json.dumps(_ExtractContext)}\n\nTweet: BABABA",
_MAX_TOKENS = 500

_s = 'abcdefghijklmnopkrstuvwxyz123456789ABCDEFGHIJKLMNOPQRSTUVWX!@£$%^&*'

async def openai_text_extraction(deep_request: str) -> Dict:
    """Calling OpenAI API with a prompt and extract
    the relevant information out of it into a dict

    :param str deep_request: A raw text to send to openai

    :return:
    """
    prompt = f"{_OPENAI_PROMPT}\"{deep_request}\""
    logging.info(f'About to query OpenAI with the folloiwng prompt: "{prompt}"')
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=_MAX_TOKENS,
        top_p=1,
        frequency_penalty=2,
        presence_penalty=0
    )
    logging.info(f'OpenAI response is: {response}')

    # Extract:
    raw_text_response = response['choices'][0]['text']
    try:
        json_data = json.loads(raw_text_response)
    except JSONDecodeError:
        fixed_raw_text = fix_broken_json(raw_text_response)
        json_data = json.loads(fixed_raw_text)

    return json_data


def fix_broken_json(data: str) -> str:
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Fix this broken JSON:\n{data}",
        temperature=1,
        max_tokens=_MAX_TOKENS,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text']
