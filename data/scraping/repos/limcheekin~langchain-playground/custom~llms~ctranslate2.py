# %% [markdown]
# # How to write a custom LLM wrapper
#
# This notebook goes over how to create a custom LLM wrapper, in case you want to use your own LLM or a different wrapper than one that is supported in LangChain.
#
# There is only one required thing that a custom LLM needs to implement:
#
# 1. A `_call` method that takes in a string, some optional stop words, and returns a string
#
# There is a second optional thing it can implement:
#
# 1. An `_identifying_params` property that is used to help with printing of this class. Should return a dictionary.
#
# Let's implement a very simple custom LLM that just returns the first N characters of the input.

# %%
"""Wrapper around CTranslate2 API."""
import logging
from typing import Any, Dict, List, Mapping, Optional
from pydantic import Extra, Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

import ctranslate2
import transformers
# %%

logger = logging.getLogger(__name__)


class Ct2Translator(LLM):
    """Wrapper around ctranslate2.Translator class.

    Example:
        .. code-block:: python

            from ctranslate2.llms import Ct2Translator
            translator = Ct2Translator(model_path="./ct2fast-flan-alpaca-xl")
    """
    model_path: str = None
    tokenizer_path: str = None
    inter_threads: int = 1          # inter_threads
    compute_type: str = "int8"
    translator: ctranslate2.Translator = None
    tokenizer: transformers.AutoTokenizer = None

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not
    explicitly specified."""

    class Config:
        """Configuration for this pydantic config."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {
            field.alias for field in cls.__fields__.values()}

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                logger.warning(
                    f"""{field_name} was transfered to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.model_path = kwargs.get("model_path")
        self.tokenizer_path = kwargs.get("tokenizer_path") or self.model_path
        self.inter_threads = kwargs.get("inter_threads", self.inter_threads)
        self.compute_type = kwargs.get("compute_type", self.compute_type)
        self.translator = ctranslate2.Translator(
            model_path=self.model_path,
            inter_threads=self.inter_threads,
            compute_type=self.compute_type
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path)

    @property
    def _llm_type(self) -> str:
        return "Ct2Translator"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        input_tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(prompt))
        results = self.translator.translate_batch(
            [input_tokens],
            **self.model_kwargs
        )
        output_tokens = results[0].hypotheses[0]
        output_text = self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(output_tokens))
        return output_text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "inter_threads": self.inter_threads,
            "compute_type": self.compute_type,
            **{"model_kwargs": self.model_kwargs},
        }
# %% [markdown]
# We can now use this as an any other LLM.
