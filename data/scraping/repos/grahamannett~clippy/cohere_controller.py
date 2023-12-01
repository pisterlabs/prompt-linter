from os import environ
from typing import Callable, Coroutine, Dict, List, Literal, Type

import cohere

from clippy import logger
from clippy.controllers.apis.cohere_controller_utils import Embeddings, Generations
from clippy.controllers.controller import Controller
from clippy.controllers.utils import allow_full_response
from clippy.dm import Database

# DEFAULT_SPECIAL_TOKENS = {13782: ["click"], 5317: ["type"]}
AVAILABLE_MODELS: List[str] = ["command", "command-nightly", "command-light", "base"]


ClientTypes: Dict[str, Callable] = {
    "async": cohere.AsyncClient,
    "sync": cohere.Client,
}


class Responses:
    """
    A class to hold the response types.
    """

    Generations: Type[Generations] = Generations
    Embeddings: Type[Embeddings] = Embeddings


class CohereController(Controller):
    """
    A class to interact with the Cohere API.
    """

    # cohere config
    class config:
        truncate: str = "END"  # one of NONE|START|END
        # model: str = "command"
        model: str = AVAILABLE_MODELS[0]
        embed_model: str = "embed-english-v2.0"

    client: cohere.AsyncClient
    client_exception: Type[cohere.error.CohereError] = cohere.error.CohereError
    client_exception_message: str = "Cohere fucked up: {0}"

    _is_async: bool = True

    def __init__(self, client: cohere.AsyncClient = None, *args, **kwargs):
        """
        Initialize the CohereController.
        """
        super().__init__(*args, **kwargs)
        self.client = client or CohereController.get_client()

    @classmethod
    def get_client(
        cls, api_key: str = None, check_api_key: bool = True, client_type: Literal["async", "sync"] = "async"
    ) -> cohere.AsyncClient | cohere.Client:
        """
        Get the Cohere client.
        """
        if client_type == "sync":
            logger.warn("USING SYNC LLM Client")

        ClientCls = ClientTypes[client_type]
        api_key = api_key or environ.get("COHERE_KEY")
        return ClientCls(api_key=api_key, check_api_key=check_api_key)

    @allow_full_response(lambda resp: resp.tokens)
    async def tokenize(self, text, model: str = config.model):
        """
        Tokenize the given text.
        """
        return await self.client.tokenize(text, model=model)

    @Database.database_api_calls
    async def embed(
        self, texts: List[str] | str, truncate: str = config.truncate, model: str = config.embed_model, **kwargs
    ) -> Embeddings:
        """
        Embed the given texts.
        """
        if isinstance(texts, str):
            texts = [texts]

        response = await self.client.embed(texts=texts, truncate=truncate, model=model, **kwargs)
        response = Embeddings(embeddings=response.embeddings, meta=response.meta)

        return response

    @Database.database_api_calls
    async def generate(
        self,
        prompt: str = None,
        model: str = config.model,
        temperature: float = 0.5,
        k: int = 0,
        num_generations: int = None,
        max_tokens: int = 20,
        stop_sequences: List[str] = None,
        return_likelihoods: str = "GENERATION",
        truncate: str = "NONE",  # must be one of NONE/START/END
        stream: bool = False,
        **kwargs,
    ) -> Generations:
        """
        Generate text using the Cohere API.
        # https://docs.cohere.com/reference/generate
        """
        response = await self.client.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            k=k,
            num_generations=num_generations,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            return_likelihoods=return_likelihoods,
            truncate=truncate,
            stream=stream,
            **kwargs,
        )

        gen_response = Generations.from_response(response)
        return gen_response

    async def close(self) -> Coroutine:
        """
        Close the Cohere client.
        """
        return await self.client.close()

    def _get_special_tokens(self, tokens: List[str] = ["click", "type"]) -> Dict[int, List[str]]:
        """
        Get the special tokens. not sure if i want to make Click/CLICK/etc as tokens as well
        """
        special_tokens_dict: Dict[int, List[str]] = {}

        for token in tokens:
            tokenized: cohere.client.Tokens = self.tokenize(token)
            special_tokens_dict[tokenized.tokens[0]] = tokenized.token_strings

        return special_tokens_dict
