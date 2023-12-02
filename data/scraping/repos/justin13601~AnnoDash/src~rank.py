import os
import time
import openai
import cohere
import json
from src.prompts import *
from functools import lru_cache


@lru_cache(maxsize=None)
def get_response(model, system_prompt, user_message):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        temperature=0,
        max_tokens=2000,
    )
    return response


def set_up_rank(method):
    if method == 'gpt':
        print("OpenAI GPT ranking selected.")
        return RankGPT()
    elif method == 'cohere':
        print("CohereAI API ranking selected.")
        return RankCohere()


def rank(target, choices, method, metadata=None, **kwargs):
    if method == 'gpt':  # GPT ranking
        # start_time = time.time()
        kwargs['ranker'].prepare_prompt(target=target, choices=choices, metadata=metadata)
        ranked_ids = json.loads(kwargs['ranker'].get_rank_results())
        # elapsed_time = time.time() - start_time
        # print('--------GPT Ranking Time:', elapsed_time, 'seconds--------')

        # Sort datatable for display
        return [choices[i] for i in ranked_ids]
    elif method == 'cohere':  # CohereAI ranking
        # start_time = time.time()
        kwargs['ranker'].prepare_ranker(target=target, choices=choices)
        ranked_desc = kwargs['ranker'].get_rank_results()
        # elapsed_time = time.time() - start_time
        # print('--------Cohere Ranking Time:', elapsed_time, 'seconds--------')

        # Sort datatable for display
        return sorted(choices, key=lambda x: ranked_desc.index(x['LABEL']))


class RankGPT:
    def __init__(self):
        # Load your API key from an environment variable or secret management service
        self.api_key = os.environ['OPENAI_API_KEY']
        self.model = 'gpt-3.5-turbo'
        # self.model = 'gpt-3.5-turbo-0314'
        # self.model = 'gpt-4-0314'
        self.system_prompt = ''
        self.user_prompt = ''

    def prepare_prompt(self, target, choices, metadata):
        keys_to_keep = ['id', 'CODE', 'LABEL']  # , 'SYSTEM', 'SCALE_TYP', 'METHOD_TYP', 'CLASS']
        filtered_choices = [{key: d[key] for key in keys_to_keep} for d in choices]

        code_text = ""
        for each_code in filtered_choices:
            code_text += f"{each_code['id']},{each_code['CODE']},{each_code['LABEL'].strip()}\n"
        self.system_prompt = system_prompt_template
        self.user_prompt = user_prompt_template.format(
            target=target,
            choices=code_text,
            examples=', '.join(metadata['examples'])
        )
        return

    def get_rank_results(self):
        openai.api_key = self.api_key
        response = get_response(self.model, self.system_prompt, self.user_prompt)
        return response['choices'][0]['message']['content']


class RankCohere:
    def __init__(self):
        # Load your API key from an environment variable or secret management service
        self.api_key = os.environ['COHERE_API_KEY']
        self.model = "rerank-multilingual-v2.0"
        self.query = ''
        self.documents = []

    def prepare_ranker(self, target, choices):
        self.query = target
        self.documents = [d['LABEL'] for d in choices]
        return

    def get_rank_results(self):
        co = cohere.Client(self.api_key)
        rerank_hits = co.rerank(query=self.query, documents=self.documents, top_n=len(self.documents), model=self.model)

        result = []
        for hit in rerank_hits:
            result.append(self.documents[hit.index])
        return result
