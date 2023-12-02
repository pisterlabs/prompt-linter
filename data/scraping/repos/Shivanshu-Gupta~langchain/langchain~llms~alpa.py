"""Wrapper around Alpa Server and APIs."""
import logging
import sys
import time
import pause
import openai

from typing import Any, Dict, Generator, List, Mapping, Optional, Tuple, Union

from pydantic import BaseModel, Extra, Field, root_validator

from langchain.llms.base import BaseLLM
from langchain.schema import Generation, LLMResult
from langchain.utils import get_from_dict_or_env
from langchain.llms.utils import enforce_stop_tokens

from rich import print

logger = logging.getLogger(__name__)

class OptModel(BaseLLM, BaseModel):
    model_name: str = "opt-30b"
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    min_tokens: int = 1
    """The minimum number of tokens to generate in the completion."""
    max_tokens: int = 256
    """The maximum number of tokens to generate in the completion.
    -1 returns as many tokens as possible given the prompt and
    the models maximal context size."""
    top_p: float = 1
    """Total probability mass of tokens to consider at each step."""
    n: int = 1
    """How many completions to generate for each prompt."""
    best_of: int = 1
    """Generates best_of completions server-side and returns the "best"."""
    echo: bool = False
    """Whether to return the prompt in addition to the completion."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    batch_size: int = 20
    """Batch size to use when passing multiple documents to generate."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.ignore

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                logger.warning(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transfered to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        normal_params = {
            "temperature": self.temperature,
            "min": self.min_tokens,
            "max": self.max_tokens,
            "top_p": self.top_p,
            "n": self.n,
            "best_of": self.best_of,
            "echo": self.echo,
        }
        return {**normal_params, **self.model_kwargs}

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        return self._default_params

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "openai"

    def get_num_tokens(self, text: str) -> int:
        """Calculate num tokens with tiktoken package."""
        raise NotImplementedError()

    def modelname_to_contextsize(self, modelname: str) -> int:
        """Calculate the maximum number of tokens possible to generate for a model.

        text-davinci-003: 4,000 tokens
        text-curie-001: 2,048 tokens
        text-babbage-001: 2,048 tokens
        text-ada-001: 2,048 tokens
        code-davinci-002: 8,000 tokens
        code-cushman-001: 2,048 tokens

        Args:
            modelname: The modelname we want to know the context size for.

        Returns:
            The maximum context size

        Example:
            .. code-block:: python

                max_tokens = openai.modelname_to_contextsize("text-davinci-003")
        """
        raise NotImplementedError()

    def max_tokens_for_prompt(self, prompt: str) -> int:
        """Calculate the maximum number of tokens possible to generate for a prompt.

        Args:
            prompt: The prompt to pass into the model.

        Returns:
            The maximum number of tokens to generate for a prompt.

        Example:
            .. code-block:: python

                max_tokens = openai.max_token_for_prompt("Tell me a joke.")
        """
        num_tokens = self.get_num_tokens(prompt)

        # get max context size for model by name
        max_size = self.modelname_to_contextsize(self.model_name)
        return max_size - num_tokens

class OptAlpaServer(OptModel):
    """Wrapper around OpenAI large language models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain import OpenAI
            openai = OpenAI(model_name="text-davinci-003")
    """

    url: str = "https://api.alpa.ai"
    client: Any  #: :meta private:
    api_key: Optional[str] = None
    """only necessary if using https://api.alpa.ai and not own server."""
    base_delay: int = 4
    next_slot: int = time.time()
    keep_trying: bool = False

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        from llm_serving.client import Client
        values["client"] = Client(values['url'], values['api_key'], default_model='opt-30b')
        return values

    def _get_response(self, prompts: List[str], params):
        """Get the response from the API."""
        import json
        params = {k: v for k, v in params.items() if k in ['min_tokens', 'max_tokens', 'temperature', 'top_p', 'echo', 'model_name']}
        try:
            response = self.client.completions(prompts, **params)
        except json.JSONDecodeError as e:
            response = self.client.completions(prompts, **params)
        return response

    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Call out to OpenAI's endpoint with k unique prompts.

        Args:
            prompts: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The full LLM output.

        Example:
            .. code-block:: python

                response = openai.generate(["Tell me a joke."])
        """
        # TODO: write a unit test for this
        params = self._invocation_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop

        sub_prompts = [
            prompts[i : i + self.batch_size]
            for i in range(0, len(prompts), self.batch_size)
        ]
        # choices = []
        choices = {i: [] for i in range(len(prompts))}
        token_usage = {}
        for i, _prompts in enumerate(sub_prompts):
            response = self._get_response(_prompts, params)
            for choice in response['choices']:
                choices[i * self.batch_size + choice['index']].append(choice)

        def enforce_stop_tokens(text, stop):
            # This is a bit hacky, but I can't figure out a better way to enforce
            # stop tokens when making calls to huggingface_hub.
            from langchain.llms.utils import enforce_stop_tokens
            return enforce_stop_tokens(text, stop) if stop is not None else text

        generations = [[Generation(
            text=enforce_stop_tokens(choice["text"], stop), generation_info=choice)
                        for choice in choices[i]]
                       for i in range(len(prompts))]
        return LLMResult(
            generations=generations, llm_output={"token_usage": token_usage}
        )
