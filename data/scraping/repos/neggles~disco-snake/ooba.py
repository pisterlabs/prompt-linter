import copy
import json
import logging
from typing import Any, List, Mapping, Optional

import aiohttp
import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from pydantic import Extra

logger = logging.getLogger(__name__)


class TextGenLLM(LLM):
    """LangChain model provider that uses oobabooga's text-generation-webui API.

    This LLM provider uses the `text-generation-webui` API to generate text.
    """

    endpoint: str = "http://localhost:5000/api/v1"

    max_new_tokens: int = 250
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 1.0
    typical_p: float = 1.0
    epsilon_cutoff: float = 0.0
    eta_cutoff: float = 0.0
    repetition_penalty: float = 1.1
    encoder_repetition_penalty: float = 1.0
    top_k: int = 40
    min_length: int = 0
    no_repeat_ngram_size: int = 0
    num_beams: int = 1
    penalty_alpha: float = 0.0
    length_penalty: float = 1.0
    early_stopping: bool = False
    seed: int = -1
    add_bos_token: bool = True
    truncation_length: int = 2048
    ban_eos_token: bool = False
    skip_special_tokens: bool = True
    stopping_strings: List[str] = []

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    async def __aenter__(self) -> "TextGenLLM":
        """Async context manager entry point. Sets up the aiohttp session."""
        self._asession = aiohttp.ClientSession(base_url=self.endpoint)
        return self

    async def __aexit__(self, *err) -> None:
        """Async context manager exit point. Closes the aiohttp session."""
        await self._asession.close()
        self._asession = None

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling the API."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "typical_p": self.typical_p,
            "repetition_penalty": self.repetition_penalty,
            "encoder_repetition_penalty": self.encoder_repetition_penalty,
            "top_k": self.top_k,
            "min_length": self.min_length,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "num_beams": self.num_beams,
            "penalty_alpha": self.penalty_alpha,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
            "seed": self.seed,
            "add_bos_token": self.add_bos_token,
            "truncation_length": self.truncation_length,
            "ban_eos_token": self.ban_eos_token,
            "skip_special_tokens": self.skip_special_tokens,
            "stopping_strings": self.stopping_strings,
            "epsilon_cutoff": self.epsilon_cutoff,
            "eta_cutoff": self.eta_cutoff,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"endpoint": self.endpoint}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        resp = requests.get(url=f"{self.endpoint}/model")
        resp.raise_for_status()
        return resp.json()["result"]

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """Call text-generation-webui's synchroneous API.

        :param prompt: Prompt to pass to the API.
        :type prompt: str
        :param stop: Optional list of stopping strings, defaults to None
        :type stop: Optional[List[str]], optional
        :return: The generated response.
        :rtype: str
        """
        # merge values from self.stopping_strings and stop
        stopping_strings = [x for x in self.stopping_strings if x != ""]
        if stop is not None:
            stopping_strings.extend([x for x in stop if x != "" and x not in stopping_strings])

        # duplicate the default payload
        payload = copy.deepcopy(self._default_params)
        # add the prompt and stopping strings
        payload.update({"prompt": prompt, "stopping_strings": stopping_strings})

        resp = requests.post(
            url=f"{self.endpoint}/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload, ensure_ascii=False),
        )
        resp.raise_for_status()
        resp.encoding = "utf-8"
        return resp.json()["results"][0]["text"]
