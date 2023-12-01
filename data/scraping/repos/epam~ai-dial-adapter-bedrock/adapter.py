import boto3

from aidial_adapter_bedrock.llm.chat_model import ChatModel, Model
from aidial_adapter_bedrock.llm.model.ai21 import AI21Adapter
from aidial_adapter_bedrock.llm.model.amazon import AmazonAdapter
from aidial_adapter_bedrock.llm.model.anthropic import AnthropicAdapter
from aidial_adapter_bedrock.llm.model.stability import StabilityAdapter
from aidial_adapter_bedrock.utils.concurrency import make_async


def count_tokens(string: str) -> int:
    """
    The number of bytes is a proxy for the number of tokens for
    models which do not provide any means to count tokens.

    Any token number estimator should satisfy the following requirements:
    1. Overestimation of number of tokens is allowed.
    It's ok to trim the chat history more than necessary.
    2. Underestimation of number of tokens is prohibited.
    It's wrong to leave the chat history as is when the trimming was actually required.
    """
    return len(string.encode("utf-8"))


async def get_bedrock_adapter(model_id: str, region: str) -> ChatModel:
    bedrock = await make_async(
        lambda _: boto3.Session().client("bedrock-runtime", region), ()
    )
    model_provider = Model.parse(model_id).provider
    match model_provider:
        case "anthropic":
            return AnthropicAdapter(bedrock, model_id)
        case "ai21":
            return AI21Adapter(bedrock, model_id, count_tokens)
        case "stability":
            return StabilityAdapter(bedrock, model_id)
        case "amazon":
            return AmazonAdapter(bedrock, model_id, count_tokens)
        case _:
            raise ValueError(f"Unknown model provider: '{model_provider}'")
