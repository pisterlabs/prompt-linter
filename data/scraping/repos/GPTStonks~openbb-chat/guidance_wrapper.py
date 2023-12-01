import guidance
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig


class GuidanceWrapper:
    """Wrapper around `guidance` to be used with HF models."""

    def __init__(
        self,
        model_id: str = "openlm-research/open_llama_3b_v2",
        tokenizer_kwargs: dict = {},
        model_kwargs: dict = {},
    ):
        """Init method.

        Args:
            model_id (`str`):
                Name of the HF model to use.
        """

        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)

        # set the default language model used to execute guidance programs
        guidance.llm = guidance.llms.Transformers(
            model=model_id, tokenizer=tokenizer, **model_kwargs
        )

    def __call__(self, *args, **kwargs):
        """Calls `guidance.__call__` passing all the arguments to it."""
        return guidance(*args, **kwargs)
