from __future__ import annotations

from typing import Any, Dict, List, Optional, Iterator, AsyncIterator, Set
import logging

from langchain.llms.base import BaseLLM
from langchain.pydantic_v1 import Field, root_validator
from langchain.schema import Generation, LLMResult
from langchain.utils import get_from_dict_or_env
from langchain.schema.output import GenerationChunk
from langchain.callbacks.manager import (CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun)

from .commons import completion_with_retry, acompletion_with_retry, response_text_format, response_handler
from http import HTTPStatus

logger = logging.getLogger(__name__)


def update_token_usage(
    keys: Set[str], response: Dict[str, Any], token_usage: Dict[str, Any]
) -> None:
    """Update token usage."""
    _keys_to_use = keys.intersection(response["usage"])
    for _key in _keys_to_use:
        if _key not in token_usage:
            token_usage[_key] = response["usage"][_key]
        else:
            token_usage[_key] += response["usage"][_key]


def _stream_response_to_generation_chunk(
    stream_response: Dict[str, Any],
) -> GenerationChunk:
    """Convert a stream response to a generation chunk."""
    return GenerationChunk(
        text=stream_response["output"]["choices"][0]["message"]["content"],
        generation_info=dict(
            finish_reason=stream_response["output"]["choices"][0].get("finish_reason", None),
        ),
    )


class BaseDashScope(BaseLLM):
    """Base DashScope large language model class."""
    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"dashscope_api_key": "DASHSCOPE_API_KEY"}

    @property
    def lc_serializable(self) -> bool:
        return True

    client: Any = None
    model_name: str = Field(default="qwen-turbo", alias="model"),
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    result_format: str = "message"
    """openai-compatible messages format"""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    top_p: float = 0.8
    """Total probability mass of tokens to consider at each step."""
    n: int = 1
    """How many completions to generate for each prompt."""
    dashscope_api_key: Optional[str] = None
    """Dashscope api key provide by alicloud."""
    streaming: bool = False
    """Whether to stream the results or not."""
    max_retries: int = 3
    """Maximum number of retries to make when generating."""
    prefix_messages: List = Field(default_factory=list)
    """Series of messages for Chat input."""

    def __new__(cls, **data: Any) -> BaseDashScope:
        return super().__new__(cls)

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        get_from_dict_or_env(values, "dashscope_api_key", "DASHSCOPE_API_KEY")
        try:
            import dashscope
        except ImportError:
            raise ImportError(
                "Could not import dashscope python package. "
                "Please install it with `pip install dashscope`."
            )
        try:
            values["client"] = dashscope.Generation
        except AttributeError:
            raise ValueError(
                "`dashscope` has no `Generation` attribute, this is likely "
                "due to an old version of the dashscope package. Try upgrading it "
                "with `pip install --upgrade dashscope`."
            )

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        normal_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "result_format": self.result_format,
        }

        return {**normal_params, **self.model_kwargs}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "qwen"
        
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        params: Dict[str, Any] = {
            **self._default_params,
            **kwargs,
            "model": self.model_name,
            "stream": True,
        }

        text_cursor = 0
        for stream_resp in completion_with_retry(self, prompt=prompt, run_manager=run_manager, **params):
            if stream_resp.status_code == HTTPStatus.OK:
                stream_resp, text_cursor = response_text_format(stream_resp, text_cursor)
                chunk = _stream_response_to_generation_chunk(stream_resp)
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        chunk=chunk,
                        verbose=self.verbose,
                    )
            else:
                logger.warning("http request failed: code: %s", stream_resp.status_code)

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        params: Dict[str, Any] = {
            # **{"model": self.model_name},
            **self._default_params,
            **kwargs,
            "model": self.model_name,
            "stream": True
        }

        text_cursor = 0
        async for stream_resp in await acompletion_with_retry(self, prompt=prompt, run_manager=run_manager, **params):
            if stream_resp.status_code == HTTPStatus.OK:
                stream_resp, text_cursor = response_text_format(stream_resp, text_cursor)
                chunk = _stream_response_to_generation_chunk(stream_resp)
                yield chunk
                if run_manager:
                    await run_manager.on_llm_new_token(
                        chunk.text,
                        chunk=chunk,
                        verbose=self.verbose,
                    )
            else:
                logger.warning("http request failed: code: %s", stream_resp.status_code)

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        choices = []
        token_usage: Dict[str, int] = {}
        _keys = {"input_tokens", "output_tokens"}
        params: Dict[str, Any] = {
            **{"model": self.model_name},
            **self._default_params,
            **kwargs,
        }
        if self.streaming:
            if len(prompts) > 1:
                raise ValueError("Cannot stream results with multiple prompts.")
            generation: Optional[GenerationChunk] = None
            for chunk in self._stream(prompts[0], stop, run_manager, **params):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            choices.append(
                {
                    "text": generation.text,
                    "finish_reason": generation.generation_info.get("finish_reason")
                }
            )
        else:
            response = completion_with_retry(
                self,
                prompt=prompts[0],
                run_manager=run_manager,
                **params,
            )

            response = response_handler(response)

            for v in response["output"]["choices"]:
                choices.append({
                    "text": v["message"]["content"],
                    "finish_reason": v["finish_reason"]
                })
            update_token_usage(_keys, response, token_usage)
        return self.create_llm_result(choices, prompts, token_usage)
    
    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompts."""
        choices = []
        token_usage: Dict[str, int] = {}
        _keys = {"input_tokens", "output_tokens"}
        params: Dict[str, Any] = {
            **{"model": self.model_name},
            **self._default_params,
            **kwargs,
        }
        if self.streaming:
            if len(prompts) > 1:
                raise ValueError("Cannot stream results with multiple prompts.")
            generation: Optional[GenerationChunk] = None
            async for chunk in self._astream(prompts[0], stop, run_manager, **params):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            choices.append(
                {
                    "text": generation.text,
                    "finish_reason": generation.generation_info.get("finish_reason")
                }
            )
        else:
            # _agenerate 已经是 async 函数了，这里走同步逻辑
            response = completion_with_retry(
                self,
                prompt=prompts[0],
                run_manager=run_manager,
                **params,
            )

            response = response_handler(response)

            for v in response["output"]["choices"]:
                choices.append({
                    "text": v["message"]["content"],
                    "finish_reason": v["finish_reason"]
                })
            update_token_usage(_keys, response, token_usage)
            result = self.create_llm_result(choices, prompts, token_usage)
            return result

    def create_llm_result(
        self, choices: Any, prompts: List[str], token_usage: Dict[str, int]
    ) -> LLMResult:
        """Create the LLMResult from the choices and prompts."""
        generations = []
        for i, _ in enumerate(prompts):
            sub_choices = choices[i * self.n: (i + 1) * self.n]
            # print(choices)
            # print(sub_choices)
            
            generations.append(
                [
                    Generation(
                        text=choice["text"],
                        generation_info=dict(
                            finish_reason=choice.get("finish_reason"),
                        ),
                    )
                    for choice in sub_choices
                ]
            )
        llm_output = {"token_usage": token_usage, "model_name": self.model_name}
        return LLMResult(generations=generations, llm_output=llm_output)


class Qwen_v1(BaseDashScope):
    def __new__(cls, **data: Any) -> Qwen_v1:
        return super().__new__(cls, **data)
