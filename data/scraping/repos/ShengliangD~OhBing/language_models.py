import logging
import openai
import zhipuai
import time
import os
import sys
import utils

logger = utils.get_logger(__name__)


class LanguageModel:
    def generate(self, prompt):
        raise NotImplementedError()

    def encode(self, inp: str):
        raise NotImplementedError()


class LanguageModelOpenAI(LanguageModel):
    def __init__(self, **kwargs):
        openai.api_base = os.environ.get(
            "OPENAI_API_BASE", "https://api.openai.com/v1")
        openai.api_key = os.environ["OPENAI_API_KEY"]

    def generate(self, prompt):
        while True:
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-4", messages=[{"role": "user", "content": prompt}])
                break
            except openai.error.RateLimitError:
                logger.warning(
                    'rate limit exceeded during generation, will try again in 10s')
                time.sleep(10)
            except openai.error.APIError:
                logger.warning('API error, will try again in 10s')
                time.sleep(10)
        return completion.choices[0].message.content

    def encode(self, inp: str):
        while True:
            try:
                embedding = openai.Embedding.create(
                    input=inp, model="text-embedding-ada-002")['data'][0]['embedding']
                break
            except openai.error.RateLimitError:
                logger.warning(
                    'rate limit exceeded during encoding, will try again in 10s')
                time.sleep(10)
            except openai.error.APIError:
                logger.warning('API error, will try again in 10s')
                time.sleep(10)
        return embedding


class LanguageModelZhipu(LanguageModel):
    def __init__(self, **kwargs):
        zhipuai.api_key = os.environ["ZHIPUAI_API_KEY"]

    def generate(self, prompt):
        while True:
            try:
                completion = zhipuai.model_api.invoke(
                    model="chatglm_pro",
                    prompt=[{"role": "user", "content": prompt}]
                )
                break
            except Exception as e:
                logger.error(str(e))
                time.sleep(10)
        return completion['data']['choices'][0]['content']

    def encode(self, inp: str):
        while True:
            try:
                embedding = zhipuai.model_api.invoke(
                    model="text_embedding",
                    prompt=inp
                )
                break
            except Exception as e:
                logger.error(str(e))
                time.sleep(10)
        return embedding['data']['embedding']


def create(name: str, *args, **kwargs) -> LanguageModel:
    return {
        'openai': LanguageModelOpenAI,
        'zhipu': LanguageModelZhipu,
    }[name](*args, **kwargs)
