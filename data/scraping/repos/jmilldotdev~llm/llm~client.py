"""llm class."""
import logging
from collections.abc import Iterable
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from llm_providers.ai21 import AI21Provider
from llm_providers.cohere import CohereProvider
from llm_providers.gooseai import GooseAIProvider
from llm_providers.openai import OpenAIProvider
from tqdm.auto import tqdm

from llm.caches.noop import NoopCache
from llm.caches.redis import RedisCache
from llm.caches.sqlite import SQLiteCache
from llm.prompt import Prompt
from llm.response import Response
from llm.session import Session

logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

CLIENT_CONSTRUCTORS = {
    "ai21": AI21Provider,
    "cohere": CohereProvider,
    "gooseai": GooseAIProvider,
    "openai": OpenAIProvider,
}

CACHE_CONSTRUCTORS = {
    "redis": RedisCache,
    "sqlite": SQLiteCache,
    "noop": NoopCache,
}


class llm:
    """llm session object."""

    def __init__(
        self,
        client_name: str = "openai",
        client_connection: Optional[str] = None,
        cache_name: str = "noop",
        cache_connection: Optional[str] = None,
        session_id: Optional[str] = None,
        stop_token: str = "",
        **kwargs: Any,
    ):
        """
        Initialize llm.

        Args:
            client_name: name of client.
            client_connection: connection string for client.
            cache_name: name of cache.
            cache_connection: connection string for cache.
            session_id: session id for user session cache.
                        None (default) means no session logging.
                        "_default" means generate new session id.
            stop_token: stop token prompt generation.
                        Can be overridden in run

        Remaining kwargs sent to client and cache.
        """
        if client_name not in CLIENT_CONSTRUCTORS:
            raise ValueError(
                f"Unknown client name: {client_name}. "
                f"Choices are {list(CLIENT_CONSTRUCTORS.keys())}",
            )
        if cache_name not in CACHE_CONSTRUCTORS:
            raise ValueError(
                f"Unknown cache name: {cache_name}. "
                f"Choices are {list(CACHE_CONSTRUCTORS.keys())}",
            )
        self.client_name = client_name
        # Must pass kwargs as dict for client "pop" methods removed used arguments
        self.cache = CACHE_CONSTRUCTORS[cache_name](  # type: ignore
            cache_connection,
            cache_args=kwargs,
        )
        self.client = CLIENT_CONSTRUCTORS[client_name](  # type: ignore
            client_connection,
            client_args=kwargs,
        )
        if session_id is not None:
            if session_id == "_default":
                session_id = None
            self.session = Session(session_id)
        else:
            self.session = None
        if len(kwargs) > 0:
            raise ValueError(f"{list(kwargs.items())} arguments are not recognized.")

        self.stop_token = stop_token

    def close(self) -> None:
        """Close the client and cache."""
        self.client.close()
        self.cache.close()

    def run(
        self,
        prompt: Union[Prompt, str],
        input: Optional[Any] = None,
        gold_choices: Optional[List[str]] = None,
        overwrite_cache: bool = False,
        stop_token: Optional[str] = None,
        return_response: bool = False,
        **kwargs: Any,
    ) -> Union[str, List[str], Response]:
        """
        Run the prompt.

        Args:
            prompt: prompt to run. If string, will cast to prompt.
            input: input to prompt.
            gold_choices: gold choices for max logit response (only HF models).
            overwrite_cache: whether to overwrite cache.
            stop_token: stop token for prompt generation.
                        Default is self.stop_token.
                        "" for no stop token.

        Returns:
            response from prompt.
        """
        if isinstance(prompt, str):
            prompt = Prompt(prompt)
        stop_token = stop_token if stop_token is not None else self.stop_token
        prompt_str = prompt(input)
        # Must pass kwargs as dict for client "pop" methods removed used arguments
        possible_request, full_kwargs = self.client.complete(prompt_str, kwargs)

        if len(kwargs) > 0:
            raise ValueError(f"{list(kwargs.items())} arguments are not recognized.")
        # Create cacke key
        cache_key = full_kwargs.copy()
        # Make query model dependent
        cache_key["client_name"] = self.client_name
        # Make query prompt dependent
        cache_key["prompt"] = prompt_str
        response_obj = self.cache.get(cache_key, overwrite_cache, possible_request)
        # Log session dictionary values
        if self.session:
            self.session.log_query(cache_key, response_obj.to_dict())
        # Extract text results
        if return_response:
            return response_obj
        else:
            return response_obj.get_response(stop_token)

    def run_batch(
        self,
        prompt: Prompt,
        input: Optional[Iterable[Any]] = None,
        gold_choices: Optional[List[str]] = None,
        overwrite_cache: bool = False,
        stop_token: Optional[str] = None,
        return_response: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Iterable[Union[str, List[str], Response]]:
        """
        Run the prompt on a batch of inputs.

        Args:
            prompt: prompt to run.
            input: batch of inputs.
            gold_choices: gold choices for max logit response (only HF models).
            overwrite_cache: whether to overwrite cache.
            stop_token: stop token for prompt generation.
                        Default is self.stop_token.
                        "" for no stop token.

        Returns:
            batch of responses.
        """
        if isinstance(prompt, str):
            raise ValueError(
                "Prompt must be a Prompt object for batch run on data. "
                "We only support strings in `llm.run`.",
            )
        if input is None:
            input = [None]
        return [
            self.run(
                prompt,
                inp,
                gold_choices,
                overwrite_cache,
                stop_token,
                return_response,
                **kwargs,
            )
            for inp in tqdm(input, desc="Running batch", disable=not verbose)
        ]

    def save_prompt(self, name: str, prompt: Prompt) -> None:
        """
        Save the prompt to the cache for long term storage.

        Args:
            name: name of prompt.
            prompt: prompt to save.
        """
        self.cache.set_key(name, prompt.serialize(), table="prompt")

    def load_prompt(self, name: str) -> Prompt:
        """
        Load the prompt from the cache.

        Args:
            name: name of prompt.

        Returns:
            Prompt saved with name.
        """
        return Prompt.deserialize(self.cache.get_key(name, table="prompt"))

    def get_last_queries(
        self,
        last_n: int = -1,
        return_raw_values: bool = False,
        stop_token: Optional[str] = None,
    ) -> List[Tuple[Any, Any]]:
        """
        Get last n queries from current session.

        If last_n is -1, return all queries. By default will only return the
        prompt text and result text unles return_raw_values is False.

        Args:
            last_n: last n queries.
            return_raw_values: whether to return raw values as dicts.
            stop_token: stop token for prompt results to be applied to all results.

        Returns:
            last n list of queries and outputs.
        """
        if self.session is None:
            raise ValueError(
                "Session was not initialized. Set `session_id` when loading llm.",
            )
        stop_token = stop_token if stop_token is not None else self.stop_token
        last_queries = self.session.get_last_queries(last_n)
        if not return_raw_values:
            last_queries = [
                (
                    query["prompt"],
                    Response.from_dict(response).get_response(stop_token),
                )  # type: ignore
                for query, response in last_queries
            ]
        return last_queries

    def open_explorer(self) -> None:
        """Open the explorer for jupyter widget."""
        # Open explorer
        # TODO: implement
        pass
