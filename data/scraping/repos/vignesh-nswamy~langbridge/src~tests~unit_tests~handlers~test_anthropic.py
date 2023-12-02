import random
from httpx import Request, Response
from typing import Literal, List

from anthropic import (
    RateLimitError,
    APITimeoutError,
    InternalServerError
)
from anthropic.types import Completion

import pytest
from unittest.mock import AsyncMock

from pydantic import BaseModel, Field

from langbridge.handlers.generation import AnthropicGenerationHandler
from langbridge.schema import GenerationResponse


@pytest.fixture
def fake_basic_handler() -> AnthropicGenerationHandler:
    handler = AnthropicGenerationHandler(
        model="claude-2",
        model_parameters={"temperature": 0, "max_tokens_to_sample": 50},
        inputs=[
            {"text": "Conduction is the only form of heat transfer.", "metadata": {"index": i}}
            for i in range(100)
        ],
        max_requests_per_minute=100,
        max_tokens_per_minute=20000
    )

    return handler


@pytest.fixture
def fake_handler_with_prompt() -> AnthropicGenerationHandler:
    handler = AnthropicGenerationHandler(
        model="claude-2",
        model_parameters={"temperature": 0, "max_tokens_to_sample": 50},
        inputs=[
            {"text": "Conduction is the only form of heat transfer.", "metadata": {"index": i}}
            for i in range(100)
        ],
        base_prompt="Answer if the statement below is True or False",
        max_requests_per_minute=100,
        max_tokens_per_minute=20000
    )

    return handler


@pytest.fixture
def fake_handler_with_response_model() -> AnthropicGenerationHandler:
    class ResponseModel(BaseModel):
        answer: Literal["True", "False"] = Field(description="Whether the statement is True or False")
        reason: str = Field(description="A detailed reason why the statement is True or False")

    handler = AnthropicGenerationHandler(
        model="claude-2",
        model_parameters={"temperature": 0, "max_tokens_to_sample": 50},
        inputs=[
            {"text": "Conduction is the only form of heat transfer.", "metadata": {"index": i}}
            for i in range(100)
        ],
        base_prompt="Answer if the statement below is True or False",
        response_model=ResponseModel,
        max_requests_per_minute=100,
        max_tokens_per_minute=20000
    )

    return handler


def test_prompt_tokens_computation(
    fake_basic_handler: AnthropicGenerationHandler,
    fake_handler_with_prompt: AnthropicGenerationHandler,
    fake_handler_with_response_model: AnthropicGenerationHandler
) -> None:
    assert fake_basic_handler.approximate_tokens == 1800
    assert fake_handler_with_prompt.approximate_tokens == 2800
    assert fake_handler_with_response_model.approximate_tokens == 23300


@pytest.mark.asyncio
async def test_execution(
    fake_handler_with_response_model: AnthropicGenerationHandler,
    monkeypatch
) -> None:
    async def mock_call_api() -> Completion:
        roll = random.random()

        if roll < 0.8:
            return Completion(
                completion="\n\nThis is a test.",
                model="claude-2",
                stop_reason="max_tokens"
            )
        elif roll < 0.9:
            raise RateLimitError(
                message="Rate limit reached",
                request=Request(
                    method="GET",
                    url="http://dummy-anthropic-generation.url"
                ),
                response=Response(
                    status_code=429
                ),
                body=None
            )

        elif roll < 0.95:
            raise APITimeoutError(
                request=Request(
                    method="GET",
                    url="http://dummy-anthropic-generation.url"
                )
            )

        else:
            raise InternalServerError(
                "Service is unavailable",
                request=Request(
                    method="GET",
                    url="http://dummy-anthropic-generation.url"
                ),
                response=Response(
                    status_code=500
                ),
                body=None
            )

    # Patch the execute method of the specific instance
    monkeypatch.setattr("langbridge.generation.AnthropicGeneration._call_api", AsyncMock(side_effect=mock_call_api))

    responses: List[GenerationResponse] = await fake_handler_with_response_model.execute()

    progress_tracker = fake_handler_with_response_model.progress_tracker

    assert progress_tracker.num_tasks_in_progress == 0
    assert progress_tracker.num_tasks_succeeded > 0
    assert progress_tracker.num_tasks_failed > 0
    assert progress_tracker.num_rate_limit_errors > 0
    assert progress_tracker.num_api_errors > 0
    assert progress_tracker.num_other_errors > 0
