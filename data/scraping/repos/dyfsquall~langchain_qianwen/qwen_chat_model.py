from __future__ import annotations

from http import HTTPStatus
import logging

from langchain.chat_models.base import BaseChatModel
from langchain.pydantic_v1 import Field, root_validator
from langchain.callbacks.manager import (
    CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
)
from langchain.utils import get_from_dict_or_env
from langchain.schema import ChatResult, ChatGeneration
from langchain.schema.output import ChatGenerationChunk
from langchain.schema.messages import (
    BaseMessage,
    AIMessageChunk,
    ChatMessageChunk,
    SystemMessageChunk,
    HumanMessageChunk,
)
from langchain.adapters.openai import (convert_dict_to_message, convert_message_to_dict)

import asyncio
from functools import partial

from .commons import (
    completion_with_retry, response_text_format, response_handler)

from typing import (Dict, Any, Optional, List,
                    Iterator, Tuple, Mapping, AsyncIterator)

logger = logging.getLogger(__name__)


def _stream_response_to_chat_generation_chunk(
    stream_response: Dict[str, Any],
) -> ChatGenerationChunk:
    """Convert a stream response to a chat generation chunk."""
    msg = stream_response["output"]["choices"][0]["message"]
    role = msg["role"]
    text = msg["content"]

    msg_chunk = None

    if role == "user":
        msg_chunk = HumanMessageChunk(content=text)
    elif role == "assistant":
        msg_chunk = AIMessageChunk(content=text)
    elif role == "system":
        msg_chunk = SystemMessageChunk(content=text)
    else:
        msg_chunk = ChatMessageChunk(content=text, role=role)

    return ChatGenerationChunk(
        message=msg_chunk,
        generation_info=dict(
            finish_reason=stream_response["output"]["choices"][0].get("finish_reason", None),
        ),
    )


class ChatQwen_v1(BaseChatModel):
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

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output.get("token_usage", {})
            for k, v in token_usage.items():
                if k in overall_token_usage:
                    overall_token_usage[k] += v
                else:
                    overall_token_usage[k] = v
        return {"token_usage": overall_token_usage, "model_name": self.model_name}

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts = self._create_message_dicts(messages, stop)
        params: Dict[str, Any] = {
            **self._default_params,
            **kwargs,
            "stream": True,
            "model": self.model_name,
        }
        text_cursor = 0
        for stream_resp in completion_with_retry(self, messages=message_dicts, run_manager=run_manager, **params):
            if stream_resp.status_code == HTTPStatus.OK:
                if stream_resp["output"]["choices"] and len(stream_resp["output"]["choices"]) == 0:
                    continue

                stream_resp, text_cursor = response_text_format(stream_resp, text_cursor)
                chat_chunk = _stream_response_to_chat_generation_chunk(stream_resp)
                yield chat_chunk
                if run_manager:
                    run_manager.on_llm_new_token(chat_chunk.message.content, chunk=chat_chunk.message)
            else:
                logger.warning("http request failed: code: %s", stream_resp.status_code)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        print("_generate... message: ", messages)
        params: Dict[str, Any] = {
            **self._default_params,
            **kwargs,
            "model": self.model_name,
        }

        if self.streaming:
            generation: Optional[ChatGenerationChunk] = None
            for chunk in self._stream(messages, stop, run_manager, **params):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return ChatResult(generations=[generation])
        else:
            message_dicts = self._create_message_dicts(messages, stop)

            response = completion_with_retry(
                self, messages=message_dicts, run_manager=run_manager, **params
            )

            response = response_handler(response)
            return self._create_chat_result(response)

    def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # TODO: Implement later
        raise NotImplementedError()

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        # TODO: Implement later
        return await asyncio.get_running_loop().run_in_executor(
            None,
            partial(self._generate, **kwargs),
            messages,
            stop,
            run_manager
        )

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]]]:
        message_dicts = [convert_message_to_dict(m) for m in messages]
        return message_dicts

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        llm_output = {}
        if response.status_code == HTTPStatus.OK:
            for res in response["output"]["choices"]:
                message = convert_dict_to_message(res["message"])
                gen = ChatGeneration(
                    message=message,
                    generation_info=dict(finish_reason=res.get("finish_reason")),
                )
                generations.append(gen)
            token_usage = response.get("usage", {})
            llm_output = {"token_usage": token_usage, "model_name": self.model_name}
        else:
            # TODO: error handling
            failed_msg = {"role": "assistant", "content": "Sorry, I don't know how to answer that."}
            message = convert_dict_to_message(failed_msg)
            gen = ChatGeneration(
                message=message,
                generation_info=dict({"finish_reason": "stop"}),
            )
            generations.append(gen)
            # logger.error("resp status err: ", response.status_code)
            llm_output = {"token_usage": {"input_tokens": 0, "output_tokens": 0}, "model_name": self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    def _get_invocation_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        return {
            "model": self.model_name,
            **super()._get_invocation_params(stop=stop),
            **self._default_params,
            **kwargs,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "qwen-chat"
