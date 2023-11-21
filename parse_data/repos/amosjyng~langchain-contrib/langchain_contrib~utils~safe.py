"""Module for safe invocation of langchain models."""
from typing import Dict, TypeVar, Union

from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate

T = TypeVar("T")


def safe_inputs(
    lc_object: Union[Chain, BasePromptTemplate], inputs: Dict[str, T]
) -> Dict[str, T]:
    """Filter for the subset of inputs that correspond to the given object."""
    if isinstance(lc_object, Chain):
        input_variables = lc_object.input_keys
    elif isinstance(lc_object, BasePromptTemplate):
        input_variables = lc_object.input_variables
    else:
        raise ValueError(f"Unknown type for {lc_object}")
    return {k: v for k, v in inputs.items() if k in input_variables}
