import json
import time
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any

import anthropic
from anthropic import AsyncAnthropic, Anthropic
from anthropic.types import Completion

import tiktoken
from pydantic import Field, validator

from .base import BaseGeneration
from langbridge.trackers import Usage, ProgressTracker
from langbridge.schema import GenerationResponse
from langbridge.parameters import AnthropicCompletion


_anthropic = Anthropic()
_async_anthropic = AsyncAnthropic()


class AnthropicGeneration(BaseGeneration):
    model_parameters: AnthropicCompletion
    prompt: str

    @validator("usage", pre=True, always=True)
    def resolve_usage(cls, v: Usage, values: Dict[str, Any]) -> Usage:
        if v: return v

        prompt_tokens = _anthropic.count_tokens(values["prompt"])

        completion_tokens = values["model_parameters"].max_tokens_to_sample

        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_cost=0,
            completion_cost=0
        )

    def _update_usage(self, response: GenerationResponse) -> None:
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        self.usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_cost=0,
            completion_cost=0
        )

    async def _call_api(self) -> Completion:
        return await _async_anthropic.completions.create(
            prompt=self.prompt,
            model=self.model,
            **self.model_parameters.dict()
        )

    async def invoke(
        self,
        retry_queue: asyncio.Queue,
        progress_tracker: ProgressTracker,
    ) -> GenerationResponse:
        self.callback_manager.on_llm_start(
            self.dict()
        )

        error = False
        try:
            response: Completion = await self._call_api()
        except anthropic.RateLimitError as re:
            error = True
            progress_tracker.time_last_rate_limit_error = time.time()
            progress_tracker.num_rate_limit_errors += 1
        except anthropic.APITimeoutError as te:
            error = True
            progress_tracker.num_other_errors += 1
        except anthropic.InternalServerError as se:
            error = True
            progress_tracker.num_api_errors += 1
        except (
            anthropic.BadRequestError,
            anthropic.NotFoundError,
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError
        ) as e:
            error = True

            if self.callback_manager:
                self.callback_manager.on_llm_error(
                    error=e,
                    run_id=self.id
                )

            self.max_attempts = 0
        except Exception as e:
            error = True

            if self.callback_manager:
                self.callback_manager.on_llm_error(
                    error=e,
                    run_id=self.id
                )

            self.max_attempts = 0

        if error:
            if self.max_attempts:
                retry_queue.put_nowait(self)
            else:
                progress_tracker.num_tasks_in_progress -= 1
                progress_tracker.num_tasks_failed += 1
        else:
            completion_tokens = _anthropic.count_tokens(response.completion)
            response: GenerationResponse = GenerationResponse(
                id=str(self.id),
                completion=response.completion,
                metadata=self.metadata,
                usage={
                    "prompt_tokens": self.usage.prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": self.usage.prompt_tokens + completion_tokens
                }
            )
            self._update_usage(response)

            progress_tracker.num_tasks_in_progress -= 1
            progress_tracker.num_tasks_succeeded += 1

            if self.callback_manager:
                self.callback_manager.on_llm_end(
                    response=response.dict(),
                    run_id=self.id
                )

            return response
