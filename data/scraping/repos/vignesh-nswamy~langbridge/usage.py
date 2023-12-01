from typing import Optional, Dict, Any

from pydantic import validator
from langfuse.model import LlmUsage
from langchain.callbacks.openai_info import get_openai_token_cost_for_model


class Usage(LlmUsage):
    """
    Represents the Usage stats of a Language Model.
    """
    prompt_cost: Optional[float]
    completion_cost: Optional[float]
    total_cost: Optional[float]

    class Config:
        validate_assignment = True

    @validator("total_tokens", always=True)
    def compute_total_tokens(cls, v: int, values: Dict[str, Any]):
        return values["prompt_tokens"] + values["completion_tokens"] if not v \
            else v

    @validator("total_cost", always=True)
    def compute_total_cost(cls, _, values: Dict[str, Any]) -> float:
        return values["prompt_cost"] + values["completion_cost"]
