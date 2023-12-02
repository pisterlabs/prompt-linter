#!/usr/bin/python3
# -*- coding: utf-8 -*-
import openai
import time
import random
import numpy as np
import logging
import codecs

logger = logging.getLogger(__name__)


class OpenAIClient():
    def __init__(self, keys_file):
        with open(keys_file) as f:
            self.keys = [i.strip() for i in f.readlines()]
        self.n_processes = len(self.keys)
        # self.n_processes = 1

    def call_api(self, prompt: str, engine: str, max_tokens=200, temperature=1,
                 stop=None, n=None, echo=False):
        result = None
        if temperature == 0:
            n = 1

        stop = stop.copy()
        for i, s in enumerate(stop):
            if '\\' in s:
                # hydra reads \n to \\n, here we decode it back to \n
                stop[i] = codecs.decode(s, 'unicode_escape')
        while result is None:
            try:
                key = random.choice(self.keys)
                result = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    api_key=key,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=n,
                    stop=stop,
                    logprobs=1,
                    echo=echo
                )
                return result
            except Exception as e:
                logger.info(f"{key}, Retry.")
                time.sleep(30)

    def extract_response(self, response):
        texts = [r['text'] for r in response['choices']]
        logprobs = [np.mean(r['logprobs']['token_logprobs']) for r in response['choices']]
        return [{"text": text, "logprob": logprob} for text, logprob in zip(texts, logprobs)]


def run_api(prompt, **kwargs):
    client = kwargs.pop('client')
    response = client.call_api(prompt=prompt, **kwargs)
    response = client.extract_response(response)
    return response
