"""Tests for ZBasePromptTemplate."""

from typing import Any, List

from langchain.prompts import PromptTemplate
from langchain.prompts.base import StringPromptValue
from langchain.prompts.chat import ChatPromptValue, SystemMessagePromptTemplate
from langchain.schema import PromptValue, SystemMessage

from langchain_contrib.prompts import (
    ChainedPromptTemplate,
    DefaultsTo,
    ZBasePromptTemplate,
    ZChatPromptTemplate,
    ZPromptTemplate,
)


class DemoPromptTemplate(ZBasePromptTemplate):
    """Demonstration of subclassing a ZBasePromptTemplate."""

    input_variables: List[str] = ["a", "b", "c"]

    @property
    def _prompt_type(self) -> str:
        return "demo"

    def _format_prompt(self, **kwargs: Any) -> PromptValue:
        """Return a demonstration of a partial prompt."""
        return StringPromptValue(text="a={a} b={b} c={c}".format(**kwargs))


def test_wrapper() -> None:
    """Test that ZBasePromptTemplate can wrap another prompt template."""
    base = PromptTemplate.from_template("a={a} b={b} c={c}")
    z_base = ZBasePromptTemplate.from_base_template(base)
    assert z_base.format(a="one", b=2, c=[3]) == "a=one b=2 c=[3]"


def test_partials() -> None:
    """Test that ZBasePromptTemplate can integrate partials of various kinds."""
    base = PromptTemplate.from_template("a={a} b={b} c={c}").partial(a="one")
    z_base = ZBasePromptTemplate.from_base_template(base).permissive_partial(b=2)
    assert z_base.format(c=[3]) == "a=one b=2 c=[3]"


def test_override() -> None:
    """Test that ZBasePromptTemplate can be used when overridden."""
    z_base = DemoPromptTemplate()
    assert z_base.format(a="one", b=2, c=[3]) == "a=one b=2 c=[3]"


def test_override_partials() -> None:
    """Test that ZBasePromptTemplate partials work when overridden."""
    z_base = DemoPromptTemplate().permissive_partial(a="one", b=2)
    assert z_base.format(c=[3]) == "a=one b=2 c=[3]"


def test_prompt_template() -> None:
    """Check that partials with PromptTemplate also work."""
    z_base = ZPromptTemplate.from_template("a={a} b={b} c={c}")
    assert isinstance(z_base, ZBasePromptTemplate)
    partial = z_base.partial(a="one").permissive_partial(b=2)
    assert partial.format(c=[3]) == "a=one b=2 c=[3]"


def test_partial_fn() -> None:
    """Check that the langchain demo of partial functions works as well."""

    def _partial_fn() -> List[int]:
        """Demonstration of a function used as a partial variable."""
        return [3]

    z_base = ZPromptTemplate.from_template("result={result}")
    partial = z_base.permissive_partial(result=_partial_fn)
    assert partial.format() == "result=[3]"


def test_default_redirect() -> None:
    """Check that the value can be grabbed from another key."""
    z_base = ZPromptTemplate.from_template("foo={foo}; also foo={bar}")
    partial = z_base.permissive_partial(bar=DefaultsTo("foo"))
    assert partial.format(foo="foo") == "foo=foo; also foo=foo"


def test_default_redirect_nested() -> None:
    """Test the docs example of nesting chained prompts."""
    product = ZPromptTemplate.from_template("I went to buy a {product}.")
    fruit = ZPromptTemplate.from_template("I ate the {fruit}.")
    chained = ChainedPromptTemplate(subprompts=[product, fruit], joiner=" ")
    partial = chained.permissive_partial(product=DefaultsTo("fruit"))
    assert partial.format(fruit="apple") == "I went to buy a apple. I ate the apple."


def test_default_redirect_override() -> None:
    """Test that key defaults can be overridden."""
    product = ZPromptTemplate.from_template("I went to buy a {product}.")
    fruit = ZPromptTemplate.from_template("I ate the {fruit}.")
    chained = ChainedPromptTemplate(subprompts=[product, fruit], joiner=" ")
    partial = chained.permissive_partial(product=DefaultsTo("fruit"))
    assert (
        partial.format(product="banana", fruit="apple")
        == "I went to buy a banana. I ate the apple."
    )


def test_chat_partials() -> None:
    """Test that ZChatPromptTemplate supports partials."""
    template = ZChatPromptTemplate.from_messages(
        [SystemMessagePromptTemplate.from_template("a={a} b={b} c={c}")]
    )
    assert isinstance(template, ZChatPromptTemplate)
    partial = template.partial(a="one").permissive_partial(b=2)
    assert partial.format_prompt(c=[3]) == ChatPromptValue(
        messages=[SystemMessage(content="a=one b=2 c=[3]")]
    )
