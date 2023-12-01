import os
from dataclasses import dataclass
from typing import Type

import pytest

from interlab.context import Context
from interlab.context.serialization import serialize_with_type
from interlab.lang_models import AnthropicModel, LangModelBase, OpenAiChatModel
from tests.testutils import strip_tree


def test_serialize_models():
    @dataclass
    class Models:
        first: AnthropicModel
        second: OpenAiChatModel

    models = Models(
        AnthropicModel(api_key="xxx"), OpenAiChatModel(api_key="xxx", api_org="yyy")
    )
    output = serialize_with_type(models)
    assert output == {
        "first": {"model": "claude-v1", "temperature": 1.0},
        "second": {"model": "gpt-3.5-turbo", "temperature": 0.7},
        "_type": "Models",
    }
    output = serialize_with_type(models.first)
    assert output == {
        "model": "claude-v1",
        "temperature": 1.0,
        "_type": "AnthropicModel",
    }


@pytest.mark.skipif(
    not all(os.getenv(key) for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]),
    reason="Requires API keys",
)
@pytest.mark.parametrize("model_cls", [AnthropicModel, OpenAiChatModel])
def test_query(model_cls: Type[LangModelBase]):
    model = model_cls()
    with Context("my_query") as c:
        output = model.query("Hello", max_tokens=10)
        assert isinstance(output, str)
        assert output

    output = strip_tree(c.to_dict())
    name = model_cls.__module__ + "." + model_cls.__qualname__
    assert isinstance(output["children"][0].pop("result"), str)
    print(output)
    assert output == {
        "_type": "Context",
        "name": "my_query",
        "children": [
            {
                "_type": "Context",
                "name": f"query interlab {model_cls.__qualname__} ({model.model})",
                "kind": "query",
                "inputs": {
                    "prompt": "Hello",
                    "conf": {
                        "class": name,
                        "model_name": model.model,
                        "temperature": model.temperature,
                        "max_tokens": 10,
                        "strip": True,
                    },
                },
            }
        ],
    }


@pytest.mark.skipif(
    not all(os.getenv(key) for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]),
    reason="Requires API keys",
)
@pytest.mark.parametrize("model", [AnthropicModel, OpenAiChatModel])
@pytest.mark.asyncio
async def test_aquery(model: Type[LangModelBase]):
    model = model()
    output = await model.aquery("Hello", max_tokens=10)
    assert isinstance(output, str)
    assert output
