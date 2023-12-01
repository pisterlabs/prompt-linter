import pytest
import guidance

from app.services.guidance_wrapper import GuidanceWrapper
from app.config import OpenAIConfig

llm_model_config = OpenAIConfig(
    type="openai",
    name="test",
    description="test",
    spec={"context_length": 100},
    llm_credentials={},
)


def test_query_success(mocker):
    mocker.patch.object(
        GuidanceWrapper,
        "_get_llm",
        return_value=guidance.llms.Mock("the output"),
    )

    handlebars = """{{#user~}}I want a response to the following query:
    {{query}}{{~/user}}{{#assistant~}}
    {{gen 'response' temperature=0.0 max_tokens=500}}{{~/assistant}}"""

    guidance_wrapper = GuidanceWrapper(
        model=llm_model_config,
        handlebars=handlebars,
        parameters={"query": "Some query"},
    )

    result = guidance_wrapper.query()

    assert isinstance(result, dict)
    assert result["response"] == "the output"


def test_query_using_truncate_function(mocker):
    mocker.patch.object(
        GuidanceWrapper,
        "_get_llm",
        return_value=guidance.llms.Mock("the output"),
    )

    handlebars = """{{#user~}}I want a response to the following query:
    {{query}}{{~/user}}{{#assistant~}}
    {{gen 'answer' temperature=0.0 max_tokens=500}}{{~/assistant}}
    {{set 'response' (truncate answer 3)}}
    """

    guidance_wrapper = GuidanceWrapper(
        model=llm_model_config,
        handlebars=handlebars,
        parameters={"query": "Some query"},
    )

    result = guidance_wrapper.query()

    assert isinstance(result, dict)
    assert result["answer"] == "the output"
    assert result["response"] == "the"


def test_query_missing_required_params(mocker):
    mocker.patch.object(
        GuidanceWrapper,
        "_get_llm",
        return_value=guidance.llms.Mock("the output"),
    )

    handlebars = """{{#user~}}I want a response to the following query:
    {{query}}{{~/user}}{{#assistant~}}
    {{gen 'response' temperature=0.0 max_tokens=500}}{{~/assistant}}"""

    guidance_wrapper = GuidanceWrapper(
        model=llm_model_config,
        handlebars=handlebars,
        parameters={"somethingelse": "Something"},
    )

    with pytest.raises(KeyError, match="Command/variable 'query' not found!"):
        result = guidance_wrapper.query()

        assert isinstance(result, dict)
        assert result["response"] == "the output"
