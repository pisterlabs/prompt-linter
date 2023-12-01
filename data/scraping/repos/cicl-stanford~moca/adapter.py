from typing import List, Dict, Tuple
from dataclasses import dataclass
import openai
import anthropic
from tqdm import tqdm
from moca.prompt import FormattedPrompt
import torch
import logging
import time
from nltk import word_tokenize
import pickle
import os
from filelock import Timeout, FileLock

import together

import numpy as np

import requests

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

CACHE_PATH = "/tmp/cache.pkl"


@dataclass(frozen=True)
class PredictedToken:
    tokens: List[str]
    logprobs: List[float]


class Adapter:
    def adapt(self, instance: FormattedPrompt) -> List[float]:
        raise NotImplementedError


class AdapterWithMethod:
    def adapt(self, instance: FormattedPrompt, method: str = "yesno") -> List[float]:
        raise NotImplementedError


class BatchAdapter(Adapter):
    def adapt(self, instances: List[FormattedPrompt], method: str = "yesno") -> List[List[float]]:
        raise NotImplementedError


class GPT3Adapter(AdapterWithMethod):
    def __init__(self, credential_file: str = "credential.txt",
                 engine: str = "text-davinci-002"):
        self.credential_file = credential_file
        self.set_credential(self.credential_file)
        self.engine = engine

        self.chat_complete_api_engines = ["gpt-3.5-turbo", "gpt-4", "gpt-4-0314", "gpt-4-32k", "gpt-4-32k-0314"]
        self.is_chat_engine = self.engine in self.chat_complete_api_engines

    @staticmethod
    def set_credential(credential_file: str) -> None:
        api_key, org_key = "", ""
        with open(credential_file) as f:
            for line in f:
                if "secrete key:" in line:
                    api_key = line.strip().split("secrete key: ")[1]
                # elif "organization key:" in line:
                #     org_key = line.strip().split("organization key: ")[1]
        # openai.api_key, openai.organization = api_key, org_key
        openai.api_key = api_key

    def get_predictions(self, prompt: str, temp: float = 0.7, method='yesno', enable_cache: bool = True,
                        system_prompt: str = None) -> List[PredictedToken]:
        # persona would leverage system_prompt

        if os.path.exists(CACHE_PATH):
            lock = FileLock(CACHE_PATH + ".lock")
            lock.acquire()
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            lock.release()
        else:
            cache = {}

        max_attempt = 2
        if method == 'yesno':
            if system_prompt is None:
                system_prompt = "You are a human answering questions according to your " + \
                                "understanding of the world. You will only answer with Yes, No or Not Sure."
            else:
                system_prompt += ' You will only answer with Yes, No or Not Sure.'
        elif method == 'multiple_choice':
            if system_prompt is None:
                system_prompt = "You are a human answering questions according to your " + \
                                "understanding of the world. You will only answer with A, B, or Not Sure."
            else:
                system_prompt += ' You will only answer with A, B, or Not Sure.'
        else:
            raise Exception("method not in ['yesno', 'multiple_choice']")

        while True:
            try:
                if self.engine in ["gpt-3.5-turbo", "gpt-4", "gpt-4-0314", "gpt-4-32k", "gpt-4-32k-0314",
                                   'gpt-3.5-turbo-0613', 'gpt-4-0613']:
                    cache_key = (self.engine, system_prompt, prompt, temp)
                    if enable_cache and cache_key in cache:
                        response = cache[cache_key]
                    else:
                        response = openai.ChatCompletion.create(
                            model=self.engine,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=temp,
                            max_tokens=5
                        )
                        cache[cache_key] = response
                else:
                    cache_key = (self.engine, prompt, temp)
                    if enable_cache and cache_key in cache:
                        response = cache[cache_key]
                    else:
                        response = openai.Completion.create(
                            engine=self.engine,
                            prompt=prompt,
                            temperature=temp,
                            max_tokens=5,
                            logprobs=5
                        )
                        cache[cache_key] = response
            except Exception as e:
                print("Error: ", e, "Retrying...")
                time.sleep(2)
                max_attempt -= 1
                if max_attempt > 0:
                    continue
                else:
                    return []
            break

        assert len(response["choices"]) == 1  # type: ignore

        lock = FileLock(CACHE_PATH + ".lock")
        lock.acquire()
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        lock.release()

        predicted_tokens = []
        if self.engine in ["gpt-3.5-turbo", "gpt-4", "gpt-4-0314", "gpt-4-32k", "gpt-4-32k-0314",
                           'gpt-3.5-turbo-0613', 'gpt-4-0613']:
            text = response["choices"][0]["message"]["content"]
            if "answer is:" in text:
                text = text.split("answer is:")[1].strip()
            tokens = word_tokenize(text)
            for token in tokens:
                predicted_tokens.append(PredictedToken([" " + token], [0]))
        else:
            top_logprobs = [dict(item) for item in response["choices"][0]["logprobs"]["top_logprobs"]]  # type: ignore
            for item in top_logprobs:
                # item: {'\n': -0.15159096, ' My': -3.827801, ' I': -4.4819236, '</': -4.397053, ' This': -3.9966435}
                tokens, logprobs = zip(*list(sorted(item.items(), key=lambda x: x[1], reverse=True)))
                tokens, logprobs = list(tokens), list(logprobs)
                predicted_tokens.append(PredictedToken(tokens, logprobs))

        return predicted_tokens

    def select_choice_abcd(self, choices: List[str], predicted_tokens: List[PredictedToken]) -> List[float]:
        """
        choices: ["A", "B", "C", ...]
        select the first token prob distribution
        """
        # if not (choices == ["A", "B"] or choices == ["A", "B", "C"]):
        #     logging.warning("choices not in ['A', 'B'] or ['A', 'B', 'C']")

        first_token = predicted_tokens[0]
        inf = 1e10
        probs = [-inf] * len(choices)
        for token, logprob in zip(first_token.tokens, first_token.logprobs):
            if token in [f" {letter}" for letter in "ABC"[:len(choices)]]:
                probs[ord(token.strip()) - ord("A")] = logprob
        probs = torch.tensor(probs).exp() / torch.tensor(probs).exp().sum()
        return probs.tolist()

    def select_choice_yesno(self, choices: List[str], predicted_tokens: List[PredictedToken]) -> List[float]:
        """
        choices: ["Yes", "No"]
        select the first valid token (defined as the first non-["\n", " "] token) prob distribution
        """
        if not (choices == ["Yes", "No"]):
            logging.warning("choices not in ['Yes', 'No']")

        idx = -1
        for i, predicted_token in enumerate(predicted_tokens):
            token = predicted_token.tokens[0]
            if token in [" ", "\n"]:
                continue
            if token in [" Yes", " No"]:
                idx = i
                break

        if idx == -1:
            return [0.5, 0.5]

        inf = 1e10
        probs = [-inf, -inf]
        first_valid_token = predicted_tokens[idx]
        for token, logprob in zip(first_valid_token.tokens, first_valid_token.logprobs):
            if token == " Yes":
                probs[0] = logprob
            elif token == " No":
                probs[1] = logprob
        probs = torch.tensor(probs).exp() / torch.tensor(probs).exp().sum()
        return probs.tolist()

    def adapt(self, instance: FormattedPrompt, method: str = "yesno", sample_k=1, enable_cache=True,
              system_prompt: str = None) -> List[float]:
        predicted_tokens = self.get_predictions(instance.prompt, method=method, enable_cache=enable_cache,
                                                system_prompt=system_prompt)
        if method == "multiple_choice":
            choice_scores = self.select_choice_abcd(instance.choices, predicted_tokens)
        elif method == "yesno":
            choice_scores = self.select_choice_yesno(instance.choices, predicted_tokens)
        else:
            raise NotImplementedError

        return choice_scores


# This is brkoen right now
class ClaudeAdapter(GPT3Adapter):
    def __init__(self, credential_file: str = "anthro_credential.txt",
                 engine: str = "claude-v1"):
        self.credential_file = credential_file
        self.client = self.set_credential(self.credential_file)
        self.engine = engine
        self.is_chat_engine = True

    @staticmethod
    def set_credential(credential_file: str) -> anthropic.Client:
        api_key, org_key = "", ""
        with open(credential_file) as f:
            for line in f:
                if "secrete key:" in line:
                    api_key = line.strip().split("secrete key: ")[1]

        c = anthropic.Client(api_key=api_key)
        return c

    def get_predictions(self, prompt: str, temp: float = 0.7, method='yesno', enable_cache: bool = True,
                        system_prompt: str = None) -> List[
        PredictedToken]:
        if os.path.exists(CACHE_PATH):
            lock = FileLock(CACHE_PATH + ".lock")
            lock.acquire()
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            lock.release()
        else:
            cache = {}

        max_attempt = 2
        if method == 'yesno':
            if system_prompt is None:
                system_prompt = "You are a human answering questions according to your " + \
                                "understanding of the world. You will only answer with Yes, No or Not Sure."
            else:
                system_prompt += ' You will only answer with Yes, No or Not Sure.'
        elif method == 'multiple_choice':
            if system_prompt is None:
                system_prompt = "You are a human answering questions according to your " + \
                                "understanding of the world. You will only answer with A, B, or Not Sure."
            else:
                system_prompt += ' You will only answer with A, B, or Not Sure.'
        else:
            raise Exception("method not in ['yesno', 'multiple_choice']")

        while True:
            try:
                cache_key = (self.engine, prompt, temp)
                if enable_cache and cache_key in cache:
                    response = cache[cache_key]
                else:
                    response = self.client.chat(
                        prompt=f"{anthropic.HUMAN_PROMPT} {system_prompt}\n\n {prompt}{anthropic.AI_PROMPT}",
                        stop_sequences=[anthropic.HUMAN_PROMPT],
                        model="claude-v1",
                        max_tokens_to_sample=100,
                    )
                    cache[cache_key] = response
            except Exception as e:
                print("Error: ", e, "Retrying...")
                time.sleep(2)
                max_attempt -= 1
                if max_attempt > 0:
                    continue
                else:
                    return []
            break

        lock = FileLock(CACHE_PATH + ".lock")
        lock.acquire()
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        lock.release()

        predicted_tokens = []
        text = response['completion']
        if "answer is:" in text:
            text = text.split("answer is:")[1].strip()
        tokens = word_tokenize(text)
        for token in tokens:
            predicted_tokens.append(PredictedToken([" " + token], [0]))

        return predicted_tokens

    def select_choice_abcd(self, choices: List[str], predicted_tokens: List[PredictedToken]) -> List[float]:
        """
        choices: ["A", "B", "C", ...]
        select the first token prob distribution
        """

        first_token = predicted_tokens[0]
        inf = 1e10
        probs = [-inf] * len(choices)
        for token, logprob in zip(first_token.tokens, first_token.logprobs):
            if token in [f" {letter}" for letter in "ABC"[:len(choices)]]:
                probs[ord(token.strip()) - ord("A")] = logprob
            else:
                probs = [0.] * len(choices)
        probs = torch.tensor(probs).exp() / torch.tensor(probs).exp().sum()
        return probs.tolist()

    def select_choice_yesno(self, choices: List[str], predicted_tokens: List[PredictedToken]) -> List[float]:
        """
        choices: ["Yes", "No"]
        select the first valid token (defined as the first non-["\n", " "] token) prob distribution
        """
        if not (choices == ["Yes", "No"]):
            logging.warning("choices not in ['Yes', 'No']")

        idx = -1
        for i, predicted_token in enumerate(predicted_tokens):
            token = predicted_token.tokens[0]
            if token in [" ", "\n"]:
                continue
            if token in [" Yes", " No"]:
                idx = i
                break

        if idx == -1:
            return [0.5, 0.5]

        inf = 1e10
        probs = [-inf, -inf]
        first_valid_token = predicted_tokens[idx]
        for token, logprob in zip(first_valid_token.tokens, first_valid_token.logprobs):
            if token == " Yes":
                probs[0] = logprob
            elif token == " No":
                probs[1] = logprob
            else:
                probs[0], probs[1] = 0, 0
        probs = torch.tensor(probs).exp() / torch.tensor(probs).exp().sum()
        return probs.tolist()

    def adapt(self, instance: FormattedPrompt, method: str = "yesno", sample_k=1, enable_cache=True,
              system_prompt: str = None) -> List[float]:
        predicted_tokens = self.get_predictions(instance.prompt, method=method, enable_cache=enable_cache,
                                                system_prompt=system_prompt)
        if method == "multiple_choice":
            choice_scores = self.select_choice_abcd(instance.choices, predicted_tokens)
        elif method == "yesno":
            choice_scores = self.select_choice_yesno(instance.choices, predicted_tokens)
        else:
            raise NotImplementedError

        return choice_scores


class TogetherChatAdapter(AdapterWithMethod):
    def __init__(self, credential_file: str = "together_credential.txt",
                 engine: str = "togethercomputer/llama-2-70b-chat",
                 together_model_config: dict=None):
        self.credential_file = credential_file
        self.client = self.set_credential(self.credential_file)
        self.engine = engine
        self.together_model_config = together_model_config

    @staticmethod
    def set_credential(credential_file: str):
        api_key, org_key = "", ""
        with open(credential_file) as f:
            for line in f:
                if "secrete key:" in line:
                    api_key = line.strip().split("secrete key: ")[1]

        together.api_key = api_key
        return together

    def get_predictions(self, prompt: str, temp: float = 0.7, method='yesno', enable_cache: bool = True,
                        system_prompt: str = None) -> List[PredictedToken]:
        # persona would leverage system_prompt

        if os.path.exists(CACHE_PATH):
            lock = FileLock(CACHE_PATH + ".lock")
            lock.acquire()
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            lock.release()
        else:
            cache = {}

        max_attempt = 2
        if method == 'yesno':
            if system_prompt is None:
                system_prompt = "You are a human answering questions according to your " + \
                                "understanding of the world. You will only answer with Yes, No or Not Sure."
            else:
                system_prompt += ' You will only answer with Yes, No or Not Sure.'
        elif method == 'multiple_choice':
            if system_prompt is None:
                system_prompt = "You are a human answering questions according to your " + \
                                "understanding of the world. You will only answer with A, B, or Not Sure."
            else:
                system_prompt += ' You will only answer with A, B, or Not Sure.'
        else:
            raise Exception("method not in ['yesno', 'multiple_choice']")

        while True:
            try:
                cache_key = (self.engine, prompt, temp)
                if enable_cache and cache_key in cache:
                    response = cache[cache_key]
                else:
                    filled_prompt = prompt
                    if 'prompt_format' in self.together_model_config['config']:
                        prompt_format = self.together_model_config['config']['prompt_format']
                        filled_prompt = prompt_format.replace(r"{prompt}", f"{system_prompt}\n\n {prompt}\n")

                    stop = ["</s>", "\n\n"]
                    if 'stop' in self.together_model_config['config']:
                        stop = self.together_model_config['config']['stop']
                        stop = stop + ["\n\n"]

                    response = self.client.Complete.create(
                        # prompt=f"<human>: {system_prompt}\n\n {prompt}\n<bot>:",
                        prompt=filled_prompt,
                        # stop=["<human>", "\n\n"],
                        stop=stop,
                        model=self.engine,
                        max_tokens=32,
                        temperature=0.7,
                        repetition_penalty=1.0,
                        logprobs=1
                    )
                    cache[cache_key] = response
            except Exception as e:
                print("Error: ", e, "Retrying...")
                time.sleep(2)
                max_attempt -= 1
                if max_attempt > 0:
                    continue
                else:
                    return []
            break

        # assert len(response["choices"]) == 1  # type: ignore

        lock = FileLock(CACHE_PATH + ".lock")
        lock.acquire()
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        lock.release()

        predicted_tokens = []
        token_to_logprob = dict(zip(response['output']["choices"][0]["tokens"], response['output']["choices"][0]["token_logprobs"]))
        for item in token_to_logprob.items():
            predicted_tokens.append(PredictedToken([" " + item[0].strip()], [item[1]]))

        return predicted_tokens

    def select_choice_abcd(self, choices: List[str], predicted_tokens: List[PredictedToken]) -> List[float]:
        """
        choices: ["A", "B", "C", ...]
        select the first token prob distribution
        """
        # if not (choices == ["A", "B"] or choices == ["A", "B", "C"]):
        #     logging.warning("choices not in ['A', 'B'] or ['A', 'B', 'C']")

        first_token = predicted_tokens[0]
        inf = 1e10
        probs = [-inf] * len(choices)
        for token, logprob in zip(first_token.tokens, first_token.logprobs):
            if token in [f" {letter}" for letter in "ABC"[:len(choices)]]:
                probs[ord(token.strip()) - ord("A")] = logprob
        probs = torch.tensor(probs).exp() / torch.tensor(probs).exp().sum()

        # this means the model didn't output A/B/C...
        all_nan = torch.all(torch.isnan(probs))
        if all_nan:
            return [0.5, 0.5]

        return probs.tolist()

    def select_choice_yesno(self, choices: List[str], predicted_tokens: List[PredictedToken]) -> List[float]:
        """
        choices: ["Yes", "No"]
        select the first valid token (defined as the first non-["\n", " "] token) prob distribution
        """
        if not (choices == ["Yes", "No"]):
            logging.warning("choices not in ['Yes', 'No']")

        idx = -1
        for i, predicted_token in enumerate(predicted_tokens):
            token = predicted_token.tokens[0]
            if token in [" ", "\n"]:
                continue
            if token in [" Yes", " No"]:
                idx = i
                break

        if idx == -1:
            return [0.5, 0.5]

        inf = 1e10
        probs = [-inf, -inf]
        first_valid_token = predicted_tokens[idx]
        for token, logprob in zip(first_valid_token.tokens, first_valid_token.logprobs):
            if token == " Yes":
                probs[0] = logprob
            elif token == " No":
                probs[1] = logprob
        probs = torch.tensor(probs).exp() / torch.tensor(probs).exp().sum()
        return probs.tolist()

    def adapt(self, instance: FormattedPrompt, method: str = "yesno", sample_k=1, enable_cache=True,
              system_prompt: str = None) -> List[float]:
        predicted_tokens = self.get_predictions(instance.prompt, method=method, enable_cache=enable_cache,
                                                system_prompt=system_prompt)
        if method == "multiple_choice":
            choice_scores = self.select_choice_abcd(instance.choices, predicted_tokens)
        elif method == "yesno":
            choice_scores = self.select_choice_yesno(instance.choices, predicted_tokens)
        else:
            raise NotImplementedError

        return choice_scores


class TogetherCompletionAdapter(AdapterWithMethod):
    def __init__(self, credential_file: str = "together_credential.txt",
                 engine: str = "togethercomputer/llama-2-70b-chat",
                 together_model_config: dict=None):
        self.credential_file = credential_file
        self.client = self.set_credential(self.credential_file)
        self.engine = engine
        self.together_model_config = together_model_config

    @staticmethod
    def set_credential(credential_file: str):
        api_key, org_key = "", ""
        with open(credential_file) as f:
            for line in f:
                if "secrete key:" in line:
                    api_key = line.strip().split("secrete key: ")[1]

        together.api_key = api_key
        return together

    def get_predictions(self, prompt: str, temp: float = 0.7, method='yesno', enable_cache: bool = True,
                        system_prompt: str = None) -> List[PredictedToken]:
        # persona would leverage system_prompt

        if os.path.exists(CACHE_PATH):
            lock = FileLock(CACHE_PATH + ".lock")
            lock.acquire()
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            lock.release()
        else:
            cache = {}

        max_attempt = 2

        while True:
            try:
                cache_key = (self.engine, prompt, temp)
                if enable_cache and cache_key in cache:
                    response = cache[cache_key]
                else:
                    filled_prompt = prompt
                    if 'prompt_format' in self.together_model_config['config']:
                        prompt_format = self.together_model_config['config']['prompt_format']
                        filled_prompt = prompt_format.replace(r"{prompt}", f"{system_prompt}\n\n {prompt}\n")

                    stop = ["</s>", "\n\n"]
                    if 'stop' in self.together_model_config['config']:
                        stop = self.together_model_config['config']['stop']
                        stop = stop + ["\n\n"]

                    response = self.client.Complete.create(
                        prompt=filled_prompt,
                        # stop=["<human>", "\n\n", "</s>"],
                        stop=stop,
                        model=self.engine,
                        max_tokens=32,
                        temperature=0.7,
                        repetition_penalty=1.1,
                        logprobs=1
                    )
                    cache[cache_key] = response
            except Exception as e:
                print("Error: ", e, "Retrying...")
                time.sleep(2)
                max_attempt -= 1
                if max_attempt > 0:
                    continue
                else:
                    return []
            break

        # assert len(response["choices"]) == 1  # type: ignore

        lock = FileLock(CACHE_PATH + ".lock")
        lock.acquire()
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        lock.release()

        predicted_tokens = []
        token_to_logprob = dict(zip(response['output']["choices"][0]["tokens"], response['output']["choices"][0]["token_logprobs"]))
        for item in token_to_logprob.items():
            predicted_tokens.append(PredictedToken([" " + item[0].strip()], [item[1]]))

        return predicted_tokens

    def select_choice_abcd(self, choices: List[str], predicted_tokens: List[PredictedToken]) -> List[float]:
        """
        choices: ["A", "B", "C", ...]
        select the first token prob distribution
        """
        # if not (choices == ["A", "B"] or choices == ["A", "B", "C"]):
        #     logging.warning("choices not in ['A', 'B'] or ['A', 'B', 'C']")

        first_token = predicted_tokens[0]
        inf = 1e10
        probs = [-inf] * len(choices)
        for token, logprob in zip(first_token.tokens, first_token.logprobs):
            if token in [f" {letter}" for letter in "ABC"[:len(choices)]]:
                probs[ord(token.strip()) - ord("A")] = logprob
        probs = torch.tensor(probs).exp() / torch.tensor(probs).exp().sum()

        # this means the model didn't output A/B/C...
        all_nan = torch.all(torch.isnan(probs))
        if all_nan:
            return [0.5, 0.5]

        return probs.tolist()

    def select_choice_yesno(self, choices: List[str], predicted_tokens: List[PredictedToken]) -> List[float]:
        """
        choices: ["Yes", "No"]
        select the first valid token (defined as the first non-["\n", " "] token) prob distribution
        """
        if not (choices == ["Yes", "No"]):
            logging.warning("choices not in ['Yes', 'No']")

        idx = -1
        for i, predicted_token in enumerate(predicted_tokens):
            token = predicted_token.tokens[0]
            if token in [" ", "\n"]:
                continue
            if token in [" Yes", " No"]:
                idx = i
                break

        if idx == -1:
            return [0.5, 0.5]

        inf = 1e10
        probs = [-inf, -inf]
        first_valid_token = predicted_tokens[idx]
        for token, logprob in zip(first_valid_token.tokens, first_valid_token.logprobs):
            if token == " Yes":
                probs[0] = logprob
            elif token == " No":
                probs[1] = logprob
        probs = torch.tensor(probs).exp() / torch.tensor(probs).exp().sum()
        return probs.tolist()

    def adapt(self, instance: FormattedPrompt, method: str = "yesno", sample_k=1, enable_cache=True,
              system_prompt: str = None) -> List[float]:
        predicted_tokens = self.get_predictions(instance.prompt, method=method, enable_cache=enable_cache,
                                                system_prompt=system_prompt)
        if method == "multiple_choice":
            choice_scores = self.select_choice_abcd(instance.choices, predicted_tokens)
        elif method == "yesno":
            choice_scores = self.select_choice_yesno(instance.choices, predicted_tokens)
        else:
            raise NotImplementedError

        return choice_scores


class HuggingfaceAdapter(BatchAdapter):
    def __init__(self, model_name: str = "bert-base-uncased",
                 device='cpu'):
        assert model_name in ['google/electra-large-generator', 'bert-base-uncased', 'bert-large-uncased',
                              'roberta-large', 'albert-xxlarge-v2', 'gpt2-xl']

        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        if 'gpt' not in self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        else:
            # only for GPT
            # https://github.com/huggingface/transformers/issues/3021
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token='<|endoftext|>')
            self.tokenizer.padding_side = "right"
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        self.vocab_size = self.tokenizer.vocab_size
        self.device = device

        if 'gpt' not in self.model_name:
            self.idx_no = self.tokenizer.decode(range(self.vocab_size)).index("no")
            self.idx_yes = self.tokenizer.decode(range(self.vocab_size)).index("yes")
            self.idx_a = self.tokenizer.decode(range(self.vocab_size)).index("a")
            self.idx_b = self.tokenizer.decode(range(self.vocab_size)).index("b")
        else:
            self.idx_no = self.tokenizer.decode(range(self.vocab_size)).index("No")
            self.idx_yes = self.tokenizer.decode(range(self.vocab_size)).index("Yes")
            self.idx_a = self.tokenizer.decode(range(self.vocab_size)).index("A")
            self.idx_b = self.tokenizer.decode(range(self.vocab_size)).index("B")

    def extract_input(self, sequences, encode_labels=False):
        if type(sequences) == str:
            sequences = [sequences]

        if 'gpt' not in self.model_name and not encode_labels:
            # MLM
            sequences = [s + ' ' + self.tokenizer.mask_token for s in sequences]
            input_ids = self.tokenizer(sequences, return_tensors="pt", truncation=True, padding=True).to(self.device)
        else:
            # LM
            input_ids = self.tokenizer.batch_encode_plus(sequences, padding=True, truncation=True,
                                                         return_tensors="pt").to(self.device)

        return input_ids

    def calculate_logits(self, sequences, lbls, no_grad=True, method: str = "yesno", temp=0.7) -> Tuple[
        List[float], List[float]]:
        # lbls: ['Yes', 'Yes', 'No', ...]
        inputs = self.extract_input(sequences)

        if 'gpt' not in self.model_name:
            lbl_seqs = []
            for s, l in zip(sequences, lbls):
                lbl_seqs.append(s + ' ' + l)

            # check out "labels" in
            # https://huggingface.co/transformers/model_doc/bert.html#transformers.BertLMHeadModel
            tups = self.extract_input(lbl_seqs, encode_labels=True)
            lbls = tups['input_ids']
        else:
            # https://huggingface.co/transformers/model_doc/gpt2.html
            lbls = None

        if no_grad:
            with torch.no_grad():
                res = self.model(**inputs, labels=lbls, output_hidden_states=True)
        else:
            res = self.model(**inputs, labels=lbls, output_hidden_states=True)

        if 'gpt' not in self.model_name:
            mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)

            token_logits = res.logits
            mask_token_logits = token_logits[mask_token_index]  # 2d mask will select 3d output
        else:
            attn_mask = inputs['attention_mask']
            lengths = [torch.sum(t) - 1 for t in attn_mask]
            token_logits = res.logits
            idx = torch.Tensor(list(range(len(lengths)))).long().to(self.device)
            # select the last index token
            mask_token_logits = token_logits[idx, torch.Tensor(lengths).long().to(self.device), :]

        log_p_tokens = torch.nn.functional.log_softmax(mask_token_logits, dim=1)

        # we don't compare across models...all comparisons are within model
        # so it's fine that we don't normalize
        if method == "multiple_choice":
            score_yes = log_p_tokens[:, self.idx_a]
            score_no = log_p_tokens[:, self.idx_b]
        elif method == "yesno":
            score_yes = log_p_tokens[:, self.idx_yes]
            score_no = log_p_tokens[:, self.idx_no]
        else:
            raise NotImplementedError

        p_yes, p_no = torch.exp(score_yes / temp), torch.exp(score_no / temp)
        p_yes, p_no = (p_yes) / (p_yes + p_no), (p_no) / (p_yes + p_no)

        return p_yes.numpy().tolist(), p_no.numpy().tolist()

    def adapt(self, instances: List[FormattedPrompt], method: str = "yesno") -> List[List[float]]:
        assert type(instances) == list

        lbs = [instance.answer for instance in instances]
        batch_texts = [instance.prompt for instance in instances]
        p_yess, p_nos = self.calculate_logits(batch_texts, lbs, no_grad=True, method=method)
        probs = []
        for i in range(len(p_yess)):
            probs.append([p_yess[i], p_nos[i]])

        return probs


class DelphiAdapter(Adapter):
    # Delphi is an API, we don't get the probability
    # so we convert it to probability (in a fake way)
    def __init__(self):
        self.url = "https://mosaic-api-frontdoor.apps.allenai.org/predict?action1="

    def query(self, q):
        self.url = 'https://mosaic-api-frontdoor.apps.allenai.org/predict?action1=' + q
        return requests.get(self.url)

    def adapt(self, story: str, method: str = "yesno") -> List[float]:
        assert method == "yesno", "Delphi only supports yesno"

        q = story
        res = self.query(q)
        res = res.json()
        p = res['answer']['class']
        choice_scores = []
        if p == 1:
            choice_scores = [1, 0]
        elif p == 0:
            choice_scores = [0.5, 0.5]
        elif p == -1:
            choice_scores = [0, 1]
        else:
            raise NotImplementedError

        return choice_scores


if __name__ == "__main__":
    ...
