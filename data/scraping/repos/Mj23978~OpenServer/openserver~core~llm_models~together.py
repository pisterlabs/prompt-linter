import together
from .base import BaseLlmModel, LLmInputInterface, BaseChat
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    AIMessageChunk,
)
from langchain.pydantic_v1 import root_validator
from langchain.load.serializable import Serializable
from langchain.chat_models.base import (
    BaseChatModel,
    _generate_from_stream,
)
from langchain.callbacks.base import Callbacks
from langchain.schema.output import LLMResult, GenerationChunk, ChatGeneration, ChatGenerationChunk, ChatResult
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import (Any, Dict, Iterator, List,
                    Optional)
from langchain.utils import get_from_dict_or_env


class TogetherModel(BaseLlmModel):
    def __init__(self, input: LLmInputInterface) -> None:
        self.client = TogetherLLM(
            together_api_key=input.api_key,
            model=input.model_name if input.model_name else "Open-Orca/Mistral-7B-OpenOrca",
            top_p=input.top_p,
            top_k=input.top_k,
            temperature=input.temperature,
            max_tokens=input.max_tokens,
            stop=input.stop,
            cache=input.cache,
            verbose=input.verbose,
            callbacks=input.callbacks,
            metadata=input.metadata,
        )  # type: ignore

    def compelete(self, prompts: List[str], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None) -> LLMResult:
        result: LLMResult = self.client.generate(
            prompts=prompts, callbacks=callbacks, metadata=metadata)
        return result

    async def acompelete(self, prompts: List[str], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None):
        result = await self.client.agenerate(prompts=prompts, metadata=metadata, callbacks=callbacks)
        return result


class ChatTogetherModel(BaseChat):
    def __init__(self, input: LLmInputInterface) -> None:
        self.client = ChatTogetherLLM(
            together_api_key=input.api_key,
            model=input.model_name if input.model_name else "Open-Orca/Mistral-7B-OpenOrca",
            top_p=input.top_p,
            top_k=input.top_k,
            temperature=input.temperature,
            max_tokens=input.max_tokens,
            stop=input.stop,
            cache=input.cache,
            verbose=input.verbose,
            callbacks=input.callbacks,
            metadata=input.metadata,
        )  # type: ignore

    def compelete(self, prompts: List[List[BaseMessage]], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None) -> LLMResult:
        result: LLMResult = self.client.generate(
            messages=prompts, callbacks=callbacks, metadata=metadata)
        return result

    async def acompelete(self, prompts: List[List[BaseMessage]], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None) -> LLMResult:
        result: LLMResult = await self.client.agenerate(messages=prompts, callbacks=callbacks, metadata=metadata)
        return result


class BaseTogether(Serializable):

    client: Any
    model: str
    max_retries: int = 6
    top_p: float = 1
    top_k: int = 25
    temperature: float = 0.5
    max_tokens: int = 512
    stop: List[str] = []
    streaming: bool = False
    together_api_key: Optional[str] = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        try:
            import g4f
        except ImportError:
            raise ImportError(
                "Could not import cohere python package. "
                "Please install it with `pip install g4f`."
            )
        else:
            together_api_key = get_from_dict_or_env(
                values, "together_api_key", "TOGETHER_API_KEY"
            )
        return values


class ChatTogetherLLM(BaseChatModel, BaseTogether):

    @property
    def _llm_type(self) -> str:
        return "together"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = ['</s>'],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult | None:

        create_kwargs = {**kwargs, "stop": stop, **self._identifying_params}

        if self.streaming:
            stream_iter = self._stream(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            return _generate_from_stream(stream_iter)
        else:
            for i in range(self.max_retries):
                try:
                    text = together.Complete.create(
                        prompt=self.base_message_to_prompt(messages),
                        **create_kwargs,
                    )
                    result_list = text['output']['choices']
                    merged_text = ""
                    for result in result_list:
                        merged_text += result['text']
                    message = AIMessage(content=merged_text)
                    return ChatResult(
                        generations=[
                            ChatGeneration(
                                message=message,
                            )
                        ]
                    )
                except Exception as e:
                    print(
                        f"Error in TogetherLLM._call: {e}, trying {i+1} of {self.max_retries}")

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = ['</s>'],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params = {**kwargs, "stop": stop, **self._identifying_params}

        for token in together.Complete.create_streaming(prompt=self.base_message_to_prompt(messages), **params):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)

    def base_message_to_prompt(self, messages: List[BaseMessage]):
        prompt = ""
        added_prefix = False
        for message in messages:
            if message.type == "system":
                prompt += f"<s> <<SYS>> {message.content} <</SYS>>\\n\\n"
                added_prefix = True
            elif message.type == "human" or message.type == "user":
                prompt += f"{'' if added_prefix == True else '<s> [INST]'} {message.content} [/INST]"
            elif message.type == "chat" or message.type == "ai" or message.type == "assisstant":
                prompt += f"{message.content} </s>"
        return prompt


class TogetherLLM(LLM, BaseTogether):

    @property
    def _llm_type(self) -> str:
        return "together"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):

        create_kwargs = {**kwargs, "stop": stop, **self._identifying_params}

        if self.streaming:
            combined_text_output = ""
            for chunk in self._stream(
                prompt=prompt,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            ):
                combined_text_output += chunk.text
            return combined_text_output
        else:
            for i in range(self.max_retries):
                try:
                    text = together.Complete.create(
                        prompt=prompt,
                        **create_kwargs,
                    )
                    result_list = text['output']['choices']
                    merged_text = ""
                    for result in result_list:
                        merged_text += result['text']
                    return merged_text
                except Exception as e:
                    print(
                        f"Error in TogetherLLM._call: {e}, trying {i+1} of {self.max_retries}")

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        params = {**kwargs, **self._identifying_params}
        for token in together.Complete.create_streaming(prompt=prompt, **params):
            chunk = GenerationChunk(text=token)
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
