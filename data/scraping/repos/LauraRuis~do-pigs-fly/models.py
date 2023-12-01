from typing import List
import numpy as np
import logging
import cohere
import openai
import time
import os

logger = logging.getLogger()


class Model:
    def _get_lm_score(self, input_text) -> float:
        """
        Uses a model to get a language modelling score (NLL) for a given input text
        :param input_text: a str text to score
        :return: perplexity
        """
        raise NotImplementedError("Language modeling not implemented for this class.")

    def get_model_score(self, labels: List) -> float:
        """
        Gets a language modeling (lm) score for input_text.
        :param labels:  a str text to score
        :return: a lm or mlm score float
        """
        return self._get_lm_score(labels)

    def get_model_completion(self, labels: list) -> str:
        """
        Gets a language modeling (lm) completion for input_text.
        :param labels:  a str text to score
        :return: a str
        """
        return self._get_lm_completion(labels)


class CohereModelWrapper(Model):

    MODELS = {
        "commandxl": "command-xlarge-beta",
        "commandxlnightly": "command-xlarge-nightly",
        "commandmedium": "command-medium-beta",
        "commandmediumnightly": "command-medium-nightly",
        "xl": "xlarge",
        "large": "large",
        "medium": "medium",
        "small": "small",
    }

    def __init__(self, model_size: str):
        assert model_size in self.MODELS, (
            "Chosen model_size=%s not available." % model_size
        )
        api_key = self._get_api_key()
        self._cohere_client = cohere.Client(api_key)
        self._model_id = self.MODELS[model_size]

    @staticmethod
    def _get_api_key():
        with open(os.path.join(os.getcwd(), "static/cohere_api_key.txt")) as infile:
            key = infile.read()
        return key

    def _get_lm_score(self, input_text):

        if isinstance(input_text, list):
            ppl = []
            for _input_text in input_text:
                prediction = self._cohere_client.generate(
                    model=self._model_id,
                    prompt=_input_text,
                    max_tokens=0,
                    temperature=1,
                    k=0,
                    p=0.75,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop_sequences=[],
                    return_likelihoods="ALL",
                )
                likelihood = prediction.generations[0].likelihood
                ppl.append(np.exp(-1.0 * likelihood))
        else:
            prediction = self._cohere_client.generate(
                model=self._model_id,
                prompt=input_text,
                max_tokens=0,
                temperature=1,
                k=0,
                p=0.75,
                frequency_penalty=0,
                presence_penalty=0,
                stop_sequences=[],
                return_likelihoods="ALL",
            )
            likelihood = prediction.generations[0].likelihood
            ppl = np.exp(-1.0 * likelihood)
        return ppl

    def _get_lm_completion(self, input_text):

        if isinstance(input_text, list):
            completions = []
            for _input_text in input_text:
                num_fail = 0
                fail = True
                while fail and num_fail < 10:
                    try:
                        prediction = self._cohere_client.generate(
                            model=self._model_id,
                            prompt=_input_text,
                            max_tokens=500,
                            temperature=0,
                            k=0,
                            p=0.75,
                            frequency_penalty=0,
                            presence_penalty=0,
                            stop_sequences=[],
                            return_likelihoods="ALL",
                        )
                        completion = prediction.generations[0]
                        completions.append(completion)
                        fail = False
                    except:
                        num_fail += 1
        else:
            raise ValueError("Wrong type of input to _get_lm_score().")
        return completions

    def to(self, device):
        logger.info(f"Cohere model is cloud-based, unable to set device to {device}")


class OpenAIModel(Model):

    MODELS = {
        "davinciinstructbeta": "davinci-instruct-beta",
        "textdavinci003": "text-davinci-003",
        "textdavinci002": "text-davinci-002",
        "textdavinci001": "text-davinci-001",
        "davinci": "davinci",
        "ada": "ada",
        "curie": "curie",
        "babbage": "babbage",
        "textbabbage001": "text-babbage-001",
        "textcurie001": "text-curie-001",
        "textada001": "text-ada-001",
        "chatgpt": "gpt-3.5-turbo",
        "gpt4": "gpt-4"
    }

    def __init__(self, model_engine: str, rate_limit=False):
        assert model_engine in self.MODELS, (
            "Chosen model_engine=%s not available." % model_engine
        )
        self._model_id = self.MODELS[model_engine]
        self._rate_limit = rate_limit
        self._set_api_key()

    def _get_lm_score(self, input_text):

        if isinstance(input_text, list):
            ppl = []
            for _input_text in input_text:
                num_fail = 0
                fail = True
                while fail and num_fail < 10:
                    try:
                        prediction = openai.Completion.create(
                            engine=self._model_id,
                            prompt=_input_text,
                            max_tokens=0,
                            logprobs=1,
                            echo=True,
                        )
                        assert prediction.choices[0].logprobs.tokens[-1] in [
                            "yes",
                            "no",
                            " no",
                            " yes",
                            "one",
                            "two",
                            " one",
                            " two",
                            "1",
                            "2",
                            " 1",
                            " 2",
                            "A",
                            "B",
                            " A",
                            " B",
                        ], "Last token is not one of binary implicature options."
                        likelihood = prediction.choices[0].logprobs.token_logprobs[-1]
                        ppl.append(np.exp(-1.0 * likelihood))
                        if self._rate_limit:
                            time.sleep(4)
                        fail = False
                    except:
                        num_fail += 1
        else:
            raise ValueError("Wrong type of input to _get_lm_score().")
        return ppl

    def _get_lm_completion(self, input_text):

        if isinstance(input_text, list):
            completions = []
            for _input_text in input_text:
                if "turbo" in self._model_id or "gpt-4" in self._model_id:
                    try:
                        prediction = openai.ChatCompletion.create(
                            model=self._model_id,
                            temperature=0,
                            max_tokens=500,
                            messages=[
                                {"role": "user", "content": _input_text},
                            ]
                        )
                        completion = prediction["choices"][0]["message"]["content"]
                    except:
                        completion = ""
                else:
                    try:
                        prediction = openai.Completion.create(
                            engine=self._model_id,
                            prompt=_input_text,
                            max_tokens=500,
                            temperature=0,
                            logprobs=1,
                            echo=True,
                        )
                        completion_tokens = prediction["usage"]["completion_tokens"]
                        completion = prediction["choices"][0]["text"].split()
                        completion = " ".join(completion[-completion_tokens:])
                    except:
                        completion = ""
                completions.append(completion)
        else:
            raise ValueError("Wrong type of input to _get_lm_score().")
        return completions

    @staticmethod
    def _set_api_key():
        with open(os.path.join(os.getcwd(), "static/openai_api_key.txt")) as infile:
            key = infile.readlines()
            openai.organization = key[0].replace("\n", "")
            openai.api_key = key[1].replace("\n", "")
        return key

    def to(self, device):
        logger.info(f"Cohere model is cloud-based, unable to set device to {device}")
