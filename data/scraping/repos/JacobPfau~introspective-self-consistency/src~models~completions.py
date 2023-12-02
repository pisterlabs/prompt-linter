from typing import List, Union

from src.models import anthropic_model, openai_model

from .base_model import BaseModel
from .utils import get_model_from_string


def generate_response_with_turns(
    model: Union[str, BaseModel],
    turns: List[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> str:

    if isinstance(model, str):
        model = get_model_from_string(model)

    if (
        model.value
        in openai_model.OpenAITextModels.list() + openai_model.OpenAIChatModels.list()
    ):
        return openai_model.generate_response_with_turns(
            turns=turns,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
    elif model in anthropic_model.AnthropicChatModels.list():
        return anthropic_model.generate_chat_completion(
            prompt_turns=turns,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
    else:
        raise ValueError(f"Invalid model: {model.value}")
