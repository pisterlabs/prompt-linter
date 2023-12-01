from abc import ABC, abstractmethod
from time import sleep
from typing import Any

import openai
import pandas as pd

from ..abstract_model import AbstractModel


class AbstractChatGPT(AbstractModel, ABC):
    def __init__(self, api_key: str, api_org: str | None, model_name="gpt-3.5-turbo-0613",
                 *args, **kwargs):
        super().__init__(*args, kwargs)
        # initialize openAI
        self.api_key = api_key
        openai.organization = api_org
        openai.api_key = api_key
        self.model_name = model_name

    @property
    @abstractmethod
    def prompt(self):
        raise NotImplementedError

    def predict(self, table: pd.DataFrame,
                query: str,
                tbl_name: str) -> list[Any] | list[None]:
        """override"""
        """wrap function to process input and predict queries
        excepts the openAI errors and return an empty list"""
        # 1. model input
        model_input = self.process_input(table, query, tbl_name)
        if model_input is None:
            """Table is too large to be processed"""
            return [None]
        pred = ['nan']
        count = 1
        while pred == ['nan']:
            if count > 10:
                self.logger.error('Too many errors, aborting.')
                raise ValueError('Too many errors, aborting.')
            # 2. predict
            pred = self.predict_input(model_input, table)
            if pred == ['nan']:
                # error occurred
                sleep(60)
                self.logger.info('sleep for 60')
                count += 1
        return pred

    def predict_input(self, model_input, table) -> list[Any]:
        try:
            content = self._predict_with_api(model_input)
        except openai.error.InvalidRequestError as e:
            # raise error because the input is too long
            self.logger.error(e)
            return [None]
        except openai.error.RateLimitError as e:
            """Too many requests to the API"""
            self.logger.error(e)
            return ['nan']
        except openai.error.APIError as e:
            """Too many requests to the API"""
            self.logger.error(e)
            return ['nan']
        except openai.error.ServiceUnavailableError as e:
            """Too many requests to the API"""
            self.logger.error(e)
            return ['nan']
        except openai.error.Timeout as e:
            self.logger.error(e)
            return ['nan']
        else:
            return content

    def _predict_with_api(self, model_input):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=self.prompt + [model_input],
            temperature=0.0,  # make it deterministic
            # max_tokens=4097, # max tokens in the generated output
            # top_p=1, ## alternative to temperature
            # frequency_penalty=0,
            # presence_penalty=0
        )
        content = self._normalize_api_output(response)
        return content

    @abstractmethod
    def _normalize_api_output(self, api_output):
        raise NotImplementedError
