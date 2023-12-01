import anthropic
import pytest

from src.models.anthropic_model import (
    AnthropicChatModels,
    AnthropicTextModels,
    format_chat_prompt,
    generate_chat_completion,
    generate_completion,
)


def test_all_anthropic_models_found():
    ours = set(AnthropicChatModels.list() + AnthropicTextModels.list())
    theirs = {"claude-v1"}
    missing = ours - theirs
    assert not missing, f"Missing models: {missing}"


@pytest.mark.parametrize("model", AnthropicTextModels.list())
def test_generate_completion(model):
    test_prompt = "\n\nHuman: Once upon a time..\n\nAssistant:"
    max_tokens = 5
    text = generate_completion(test_prompt, model=model, max_tokens=max_tokens)
    tokens = anthropic.get_tokenizer().encode(text).ids
    assert len(tokens) == max_tokens
    assert (
        len(set(tokens)) == max_tokens
    )  # sanity check: assume all tokens are unique (reasonable for short text)


@pytest.mark.parametrize(
    "prompt_turns, chat_format",
    [
        ([{"role": "Human", "content": "X"}], "\n\nHuman: X\n\nAssistant:"),
        (
            [
                {"role": "Human", "content": "X"},
                {"role": "Assistant", "content": "Y"},
                {"role": "Human", "content": "Z"},
            ],
            "\n\nHuman: X\n\nAssistant: Y\n\nHuman: Z\n\nAssistant:",
        ),
    ],
)
def test_format_chat_prompt(prompt_turns, chat_format):
    assert format_chat_prompt(prompt_turns) == chat_format


@pytest.mark.parametrize("model", AnthropicChatModels.list())
def test_generate_chat_completion(model):
    max_tokens = 5
    test_prompt = [
        {"role": "Human", "content": "Hello, how are you?"},
    ]
    text = generate_chat_completion(test_prompt, model=model, max_tokens=max_tokens)
    tokens = anthropic.get_tokenizer().encode(text).ids
    assert len(tokens) == max_tokens
    assert (
        len(set(tokens)) == max_tokens
    )  # sanity check: assume all tokens are unique (reasonable for short text)
