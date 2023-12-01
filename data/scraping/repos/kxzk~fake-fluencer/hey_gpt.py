#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
from typing import Tuple

import openai

from config import configs

openai.api_key = configs["openai_key"]


def hey_gpt(subreddit: str, keyword: str) -> Tuple[str, str]:
    prompt = f"""
    Can you provide an example Instagram post related to the subreddit r/{subreddit}.
    The img_description should contain the keyword {keyword} and be described in
    and artistic fashion. The img_caption should be humorous.

    In the following format:

    img_description: <description>
    img_caption: <caption>
    """

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
    )

    output = completion.choices[0].message["content"]

    pattern = r"img_description:\s*(.*?)\s*img_caption:\s*(.*?)\s*$"
    match = re.search(pattern, output, re.DOTALL | re.MULTILINE)
    img_desc = match.group(1)
    img_caption = match.group(2)

    return (img_desc, img_caption)
