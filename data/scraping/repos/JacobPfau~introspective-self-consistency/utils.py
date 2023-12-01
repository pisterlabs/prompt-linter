from src.models.anthropic_model import AnthropicChatModels, AnthropicTextModels
from src.models.base_model import BaseModel
from src.models.openai_model import OpenAIChatModels, OpenAITextModels


def get_model_from_string(model_name: str) -> BaseModel:
    if model_name in OpenAITextModels.list():
        return OpenAITextModels(model_name)
    elif model_name in OpenAIChatModels.list():
        return OpenAIChatModels(model_name)
    elif model_name in AnthropicChatModels.list():
        return AnthropicChatModels(model_name)
    elif model_name in AnthropicTextModels.list():
        return AnthropicTextModels(model_name)
    else:
        raise KeyError(f"Invalid model name: {model_name}")
