from typing import (Any, AsyncIterator, Dict, Iterator, List,
                    Optional, Union)

from g4f import ChatCompletion, Provider
from g4f.models import Model
from g4f.Provider.base_provider import BaseProvider
from langchain.callbacks.manager import (AsyncCallbackManagerForLLMRun,
                                         CallbackManagerForLLMRun)
from langchain.llms.base import LLM
from pydantic import Field


from typing import List

from .base import BaseChat, BaseLlmModel, LLmInputInterface

from langchain.schema.output import LLMResult, GenerationChunk, ChatGeneration, ChatGenerationChunk, ChatResult
from langchain.llms.utils import enforce_stop_tokens
from langchain.callbacks.base import Callbacks
from langchain.chat_models.base import (
    BaseChatModel,
    _agenerate_from_stream,
    _generate_from_stream,
)
from langchain.pydantic_v1 import root_validator
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    AIMessageChunk
)


class G4FModel(BaseLlmModel):
    def __init__(self, input: LLmInputInterface) -> None:
        self.client = G4FLLM(
            model=input.model_name if input.model_name else "gpt-3.5-turbo",
            max_retries=input.max_retries,
            cache=input.cache,
            verbose=input.verbose,
            streaming=input.stream,
            metadata=input.metadata,
            callbacks=input.callbacks,
        )  # type: ignore

    def compelete(self, prompts: List[str], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None) -> LLMResult:
        result = self.client.generate(prompts=prompts, metadata=metadata, callbacks=callbacks)
        return result

    async def acompelete(self, prompts: List[str], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None):
        result = await self.client.agenerate(prompts=prompts, metadata=metadata, callbacks=callbacks)
        return result

class ChatG4FModel(BaseChat):
    def __init__(self, input: LLmInputInterface) -> None:
        self.client = ChatG4FLLM(
            model=input.model_name if input.model_name else "gpt-3.5-turbo",
            max_retries=input.max_retries,
            cache=input.cache,
            verbose=input.verbose,
            streaming=input.stream,
            metadata=input.metadata,
            callbacks=input.callbacks,
        )  # type: ignore

    def compelete(self, prompts: List[List[BaseMessage]], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None) -> LLMResult:
        result: LLMResult = self.client.generate(
            messages=prompts, callbacks=callbacks, metadata=metadata)
        return result

    async def acompelete(self, prompts: List[List[BaseMessage]], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None) -> LLMResult:
        result: LLMResult = await self.client.agenerate(messages=prompts, callbacks=callbacks, metadata=metadata)
        return result


class ChatG4FLLM(BaseChatModel):

    client: Any
    model: Union[Model, str] = "gpt-3.5-turbo"
    max_retries: int = 6
    prefix_messages: List = Field(default_factory=list)
    streaming: bool = False
    # provider: Optional[type[BaseProvider]] = Provider.Bard
    auth: str | None = None
    create_kwargs: dict[str, Any] = {}

    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            # "provider": self.provider,
            "auth": self.auth,
            "create_kwargs": self.create_kwargs,
            "max_retries": self.max_retries,
            "prefix_messages": self.prefix_messages,
            "stream": self.streaming,
        }

    @property
    def _llm_type(self) -> str:
        return "g4f"


    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    )  -> ChatResult | None:

        create_kwargs = {**kwargs, **self.create_kwargs}
        create_kwargs["model"] = self.model
        # if self.provider is not None:
        #     create_kwargs["provider"] = self.provider
        if self.auth is not None:
            create_kwargs["auth"] = self.auth

        if self.streaming:
            stream_iter = self._stream(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                **create_kwargs,
            )
            return _generate_from_stream(stream_iter)
        else:
            for i in range(self.max_retries):
                try:
                    text = ChatCompletion.create(
                        messages=self.base_message_to_list(messages),
                        **create_kwargs,
                    )

                    text = text if type(text) is str else "".join(text)
                    # if text:
                    message = AIMessage(content=text)
                    return ChatResult(
                        generations=[
                            ChatGeneration(
                                message=message,
                            )
                        ]
                    )
                except Exception as e:
                    print(
                        f"Error in G4FLLM._call: {e}, trying {i+1} of {self.max_retries}")

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult | None:

        create_kwargs = {**kwargs, **self.create_kwargs}
        create_kwargs["model"] = self.model
        # if self.provider is not None:
        #     create_kwargs["provider"] = self.provider
        if self.auth is not None:
            create_kwargs["auth"] = self.auth

        if self.streaming:
            stream_iter = self._astream(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                **create_kwargs,
            )
            return await _agenerate_from_stream(stream_iter)
        else:
            for i in range(self.max_retries):
                try:
                    text = ChatCompletion.create(
                        messages=self.base_message_to_list(messages),
                        **create_kwargs,
                    )

                    text = text if type(text) is str else "".join(text)
                    # if text:
                    message = AIMessage(content=text)
                    return ChatResult(
                        generations=[
                            ChatGeneration(
                                message=message,
                            )
                        ]
                    )
                except Exception as e:
                    print(
                        f"Error in G4FLLM._call: {e}, trying {i+1} of {self.max_retries}")


    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params = {**kwargs, "stream": True}
        response = ChatCompletion.create(
            messages=self.base_message_to_list(messages),
            **params
        )
        for token in response:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        params = {**kwargs, "stream": True}
        response = ChatCompletion.create(
            messages=self.base_message_to_list(messages),
            **params
        )
        for token in response:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(token, chunk=chunk)

    def base_message_to_list(self, messages: List[BaseMessage]):
        return list(map(lambda x: {"role": x.type, "content": x.content}, messages))

class G4FLLM(LLM):

    client: Any
    model: Union[Model, str] = "gpt-3.5-turbo"
    max_retries: int = 6
    prefix_messages: List = Field(default_factory=list)
    streaming: bool = False
    # provider: Optional[type[BaseProvider]] = Provider.Bard
    auth: str | None = None
    create_kwargs: dict[str, Any] = {}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            # "provider": self.provider,
            "auth": self.auth,
            "create_kwargs": self.create_kwargs,
            "max_retries": self.max_retries,
            "prefix_messages": self.prefix_messages,
            "stream": self.streaming,
        }

    @property
    def _llm_type(self) -> str:
        return "g4f"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):

        create_kwargs = {**kwargs, **self.create_kwargs}
        create_kwargs["model"] = self.model
        # if self.provider is not None:
        #     create_kwargs["provider"] = self.provider
        if self.auth is not None:
            create_kwargs["auth"] = self.auth

        if self.streaming:
            combined_text_output = ""
            for chunk in self._stream(
                prompt=prompt,
                stop=stop,
                run_manager=run_manager,
                **create_kwargs,
            ):
                combined_text_output += chunk.text
            return combined_text_output
        else:
            for i in range(self.max_retries):
                try:
                    text = ChatCompletion.create(
                        messages=[{"role": "user", "content": prompt}],
                        **create_kwargs,
                    )
                    text = text if type(text) is str else "".join(text)
                    if stop is not None:
                        text = enforce_stop_tokens(text, stop)
                    if text:
                        return text
                    print(
                        f"Empty response, trying {i+1} of {self.max_retries}")
                except Exception as e:
                    print(
                        f"Error in G4FLLM._call: {e}, trying {i+1} of {self.max_retries}")

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[AsyncCallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        create_kwargs = {**kwargs, **self.create_kwargs}
        create_kwargs["model"] = self.model
        # if self.provider is not None:
        #     create_kwargs["provider"] = self.provider
        if self.auth is not None:
            create_kwargs["auth"] = self.auth

        if self.streaming:
            combined_text_output = ""
            async for chunk in self._astream(
                prompt=prompt,
                stop=stop,
                run_manager=run_manager,
                **create_kwargs,
            ):
                combined_text_output += chunk.text
            return combined_text_output
        else:
            for i in range(self.max_retries):
                try:
                    text = await ChatCompletion.create_async(
                        messages=[{"role": "user", "content": prompt}],
                        **create_kwargs,
                    )
                    text = text if type(text) is str else "".join(text)
                    if stop is not None:
                        text = enforce_stop_tokens(text, stop)
                    if text:
                        return text
                    print(
                        f"Empty response, trying {i+1} of {self.max_retries}")
                except Exception as e:
                    print(
                        f"Error in G4FLLM._call: {e}, trying {i+1} of {self.max_retries}")
            return ""


    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        params = {**kwargs, "stream": True}
        for token in generate(prompt, params):
            chunk = GenerationChunk(text=token)
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        params = {**kwargs, "stream": True}
        for token in generate(prompt, params):
            chunk = GenerationChunk(text=token)
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(token, chunk=chunk)

def generate(prompt: str, args: dict[str, Any]):
    response = ChatCompletion.create(
            messages=[{"role": "user", "content": prompt}],
            **args
        )
    for token in response:
        yield token
