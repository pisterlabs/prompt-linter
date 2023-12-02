# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # api
#
# > 

# %%
#| default_exp api

# %%
#| hide
from nbdev.showdoc import *

# %%
#| export
import json
from functools import partial
from typing import Any
from typing import Callable
from typing import Literal

import boto3
import cohere
import numpy as np
import openai

from emblem.core import A
from emblem.core import EmbeddingConfig
from emblem.core import EmbeddingModel
from emblem.core import env_or_raise


# %%
#| export
def _fetch_and_extract(
        text: str,
        name: str,
        fetch: Callable, 
        extract: Callable,
        type: Literal["np", "tch"] | None = None
    ) -> A:
    
    try:
        response = fetch(text)
        # print(response)
    except ConnectionError:
        raise ValueError(f"Network connection error occurred while contacting {name} API.")
    except TimeoutError:
        raise ValueError(f"Connection timeout error occurred while contacting {name} API.")
    except:
        raise ValueError(f"Unexpected error occurred while contacting {name} API.")
    
    try:
        value = extract(response)
    except Exception as e:
        raise ValueError(f"Invalid response structure received from {name} API. Error: {str(e)}")

    if type is None:
        result = value
    if type == "np":
        result = np.array(value, dtype=np.float32)
    if type == "tch":
        result = torch.Tensor(value, dtype=tch.float32)

    return result


# %%
#| export
class OpenAIModel(EmbeddingModel):
    def __init__(self, config: EmbeddingConfig | None = None):
        super().__init__(config)
        openai.key = self.config.key

    def _get_default_config(self) -> EmbeddingConfig:
        default_config = EmbeddingConfig(
            key=env_or_raise("OPENAI_API_KEY"),
            name="text-embedding-ada-002",
            rate_period=60,
            rate_calls=3_000,
            rate_tokens=1_000_000
        )

        return default_config
        
    def embed(self, text: str, type: Literal["np", "tch"] | None = None) -> A:
        fetch = lambda text: openai.Embedding.create(input=text, model=self.config.name)
        extract = lambda response: response['data'][0]['embedding']
        embedding = _fetch_and_extract(text=text, name="OpenAI", fetch=fetch, extract=extract, type=type)

        return embedding

    async def embeda(self, text: str) -> A:
        # TODO: Implement with HTTPX or similar
        raise NotImplementedError


# %%
def test_model(
        model: EmbeddingModel, 
        text: str = "The brown fox jumped over the lazy dog.", 
        n: int = 5
    ) -> A:
    return model.embed(text, type="np")[:n]


# %%
test_model(OpenAIModel())


# %%
#| export
class CohereModel(EmbeddingModel):
    def __init__(self, config: EmbeddingConfig | None = None):
        super().__init__(config)
        self.client = None
        
    def _get_default_config(self) -> EmbeddingConfig:
        # Trial key rate limit
        default_config = EmbeddingConfig(
            key=env_or_raise("COHERE_API_KEY"),
            name="embed-english-light-v2.0",
            rate_period=60,
            rate_calls=100,
            rate_tokens=None
        )

        return default_config
        
    def embed(self, text: str, type: Literal["np", "tch"] | None = None) -> A:
        if self.client is None or not isinstance(self.client, cohere.Client):
            self.client = cohere.Client(self.config.key)

        fetch = lambda text: self.client.embed([text])
        extract = lambda response: response.embeddings[0]
        embedding = _fetch_and_extract(text=text, name="Cohere", fetch=fetch, extract=extract, type=type)

        return embedding

    async def embeda(text: str) -> A:
        # TODO: Implement with AsyncClient
       raise NotImplementedError


# %%
test_model(CohereModel())


# %%
#| export
class BedrockModel(EmbeddingModel):
    def __init__(self, config: EmbeddingConfig | None = None):
        super().__init__(config)
        self.client = boto3.client(service_name='bedrock-runtime', region_name=self.config.region)
    
    def _get_default_config(self) -> EmbeddingConfig:
        default_config = EmbeddingConfig(
            name="amazon.titan-embed-text-v1",
            region="us-east-1"
        )

        return default_config

    def embed(self, text: str, type: Literal["np", "tch"] | None = None) -> A:
        fetch = lambda text: self.client.invoke_model(
            accept='application/json',
            body=json.dumps({"inputText": text}),
            contentType='application/json',
            modelId=self.config.name
        )
        extract = lambda response: json.loads(response.get('body').read())["embedding"]
        embedding = _fetch_and_extract(text=text, name="Bedrock", fetch=fetch, extract=extract, type=type)

        return embedding

    async def embeda(text: str) -> A:
        raise NotImplementedError


# %%
test_model(BedrockModel())

# %%
#| hide
import nbdev

nbdev.nbdev_export()

# %%
