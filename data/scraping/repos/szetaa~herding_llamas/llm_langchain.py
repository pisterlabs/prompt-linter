from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import json
import requests
import os

# assuming server with custom API (llm_api project) started.
URI = "http://localhost:8090/api/v1/infer"


class LlmLangchain(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        stop_words = ["Observation", "\nObservation:", "\nObservations:"]
        if isinstance(stop, list) and len(stop) > 0:
            stop_words = stop_words + stop

        param = {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "early_stopping": True,
            "stop_words": stop_words,
            "do_sample": True,
            "top_p": 0.5,
            "typical_p": 1,
            "repetition_penalty": 1.18,
            "top_k": 50,
            "min_length": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": 1,
            "penalty_alpha": 0,
            "length_penalty": 1,
            "skip_special_tokens": True,  # for tokenizer
        }

        payload = {
            "prompt_key": "llama_2_langchain",
            "raw_input": prompt,
            "param": param,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('HERDING_LLAMAS_USER_TOKEN')}",
        }

        response = requests.post(URI, headers=headers, data=json.dumps(payload))

        response.raise_for_status()
        return response.json()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}
