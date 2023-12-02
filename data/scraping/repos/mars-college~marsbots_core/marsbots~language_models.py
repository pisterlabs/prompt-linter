import asyncio
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any
from typing import List

import cohere
import numpy as np
import openai
import requests

from marsbots import config
from marsbots.util import cosine_similarity


class LanguageModel(ABC):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def completion_handler(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError


@dataclass
class OpenAIGPT3LanguageModelSettings:
    engine: str = "text-davinci-002"
    temperature: float = 1.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class OpenAIGPT3LanguageModel(LanguageModel):
    def __init__(
        self,
        model_name: str = "openai-gpt3",
        api_key: str = config.LM_OPENAI_API_KEY,
        **kwargs,
    ) -> None:
        self.settings = OpenAIGPT3LanguageModelSettings(**kwargs)
        openai.api_key = api_key
        super().__init__(model_name)

    def completion_handler(
        self,
        prompt: str,
        max_tokens: int,
        stop: list = None,
        **kwargs: any,
    ) -> str:
        completion = openai.Completion.create(
            engine=self.settings.engine,
            prompt=prompt,
            max_tokens=max_tokens,
            stop=stop,
            temperature=kwargs.get("temperature") or self.settings.temperature,
            top_p=kwargs.get("top_p") or self.settings.top_p,
            frequency_penalty=kwargs.get("frequency_penalty")
            or self.settings.frequency_penalty,
            presence_penalty=kwargs.get("presence_penalty")
            or self.settings.presence_penalty,
        )
        completion_text = completion.choices[0].text
        return completion_text

    @staticmethod
    def content_safe(query: str) -> bool:
        # https://beta.openai.com/docs/engines/content-filter
        response = openai.Completion.create(
            engine="content-filter-alpha",
            prompt="<|endoftext|>" + query + "\n--\nLabel:",
            temperature=0,
            max_tokens=1,
            top_p=0,
            logprobs=10,
        )
        output_label = response["choices"][0]["text"]
        toxic_threshold = -0.355
        if output_label == "2":
            logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]
            if logprobs["2"] < toxic_threshold:
                logprob_0 = logprobs.get("0", None)
                logprob_1 = logprobs.get("1", None)
                if logprob_0 is not None and logprob_1 is not None:
                    if logprob_0 >= logprob_1:
                        output_label = "0"
                    else:
                        output_label = "1"
                elif logprob_0 is not None:
                    output_label = "0"
                elif logprob_1 is not None:
                    output_label = "1"
        if output_label not in ["0", "1", "2"]:
            output_label = "2"
        return output_label != "2"

    def document_search(
        self,
        query: str,
        documents: List[str] = None,
        file=None,
        **kwargs,
    ):
        engine = kwargs.get("engine") or self.settings.engine
        search = openai.Engine(engine).search(
            documents=documents,
            query=query,
            file=file,
        )
        return search

    def document_similarity(self, document: str, query: str, **kwargs):
        engine = kwargs.get("engine") or self.settings.engine
        doc_engine = f"text-search-{engine}-doc-001"
        query_engine = f"text-search-{engine}-query-001"
        document_embedding = self._get_embedding(document, engine=doc_engine)
        query_embedding = self._get_embedding(query, engine=query_engine)
        similarity = cosine_similarity(document_embedding, query_embedding)
        return similarity

    def most_similar_doc_idx(self, document_search_result: dict):
        return np.argmax([d["score"] for d in document_search_result["data"]])

    def _get_embedding(self, text: str, engine: str):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], engine=engine)["data"][0][
            "embedding"
        ]

    def upload_doc(self, document_path: str, purpose: str = "search"):
        openai.File.create(file=document_path, purpose=purpose)


@dataclass
class AI21JurassicLanguageModelSettings:
    model_type: str = "j1-jumbo"
    temperature: float = 1.0
    top_p: float = 1.0


class AI21JurassicLanguageModel(LanguageModel):
    def __init__(
        self,
        model_name: str = "ai21-jurassic",
        api_key: str = config.LM_AI21_API_KEY,
        **kwargs,
    ) -> None:
        self.settings = AI21JurassicLanguageModelSettings(**kwargs)
        self.api_key = api_key
        super().__init__(model_name)

    @property
    def api_url(self) -> str:
        return f"https://api.ai21.com/studio/v1/{self.settings.model_type}/complete"

    def completion_handler(
        self,
        prompt: str,
        max_tokens: int,
        stop: list = None,
        **kwargs: any,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "maxTokens": max_tokens,
            "temperature": kwargs.get("temperature") or self.settings.temperature,
            "topP": kwargs.get("top_p") or self.settings.top_p,
            "stopSequences": stop if stop else [],
        }
        response = requests.post(self.api_url, json=payload, headers=headers)
        completion = response.json()
        completion_text = completion["completions"][0]["data"]["text"]
        return completion_text


@dataclass
class CohereLanguageModelSettings:
    model_type: str = "large"
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: float = 0


class CohereLanguageModel(LanguageModel):
    def __init__(
        self,
        model_name: str = "cohere",
        api_key: str = config.LM_COHERE_API_KEY,
        **kwargs,
    ):
        self.client = cohere.Client(api_key)
        self.settings = CohereLanguageModelSettings(**kwargs)
        super().__init__(model_name)

    def completion_handler(self, prompt: str, max_tokens: int, **kwargs: any) -> str:
        prediction = self.client.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            model=kwargs.get("model_type") or self.settings.model_type,
            temperature=kwargs.get("temperature") or self.settings.model_type,
            k=kwargs.get("top_k") or self.settings.top_k,
            p=kwargs.get("top_p") or self.settings.top_p,
        )
        completion = prediction.generations[0].text
        return completion


@dataclass
class GooseAILanguageModelSettings:
    engine: str = "gpt-neo-20b"
    temperature: float = 1.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class GooseAILanguageModel(LanguageModel):
    def __init__(
        self,
        model_name: str = "gooseai",
        api_key: str = config.LM_GOOSEAI_API_KEY,
        **kwargs,
    ) -> None:
        self.settings = GooseAILanguageModelSettings(**kwargs)
        openai.api_key = api_key
        openai.api_base = "https://api.goose.ai/v1"
        super().__init__(model_name)

    def completion_handler(
        self,
        prompt: str,
        max_tokens: int,
        stop: list = None,
        **kwargs: any,
    ) -> str:
        completion = openai.Completion.create(
            engine=self.settings.engine,
            prompt=prompt,
            max_tokens=max_tokens,
            stop=stop,
            temperature=kwargs.get("temperature") or self.settings.temperature,
            top_p=kwargs.get("top_p") or self.settings.top_p,
            frequency_penalty=kwargs.get("frequency_penalty")
            or self.settings.frequency_penalty,
            presence_penalty=kwargs.get("presence_penalty")
            or self.settings.presence_penalty,
        )
        completion_text = completion.choices[0].text
        return completion_text


async def complete_text(
    language_model: LanguageModel,
    prompt: str,
    max_tokens: int,
    use_content_filter: bool = False,
    **kwargs: any,
) -> str:
    loop = asyncio.get_running_loop()
    response_safe, max_tries, num_tries = False, 3, 0
    while num_tries < max_tries and not response_safe:
        completion_text = await loop.run_in_executor(
            None,
            partial(
                language_model.completion_handler,
                prompt=prompt,
                max_tokens=int(max_tokens),
                **kwargs,
            ),
        )
        num_tries += 1
        if (
            OpenAIGPT3LanguageModel.content_safe(completion_text)
            or not use_content_filter
        ):
            response_safe = True
        else:
            print(f"Completion flagged unsafe: {completion_text}")
    if not response_safe:
        completion_text = "Sorry, try talking about something else."
    return completion_text
