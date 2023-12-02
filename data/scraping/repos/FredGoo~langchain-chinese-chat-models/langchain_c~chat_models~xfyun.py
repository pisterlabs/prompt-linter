"""Xfyun chat wrapper."""
from __future__ import annotations

import _thread as thread
import base64
import datetime
import hashlib
import hmac
import json
import logging
import ssl
import sys
from datetime import datetime
from time import mktime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)
from urllib.parse import urlencode
from urllib.parse import urlparse
from wsgiref.handlers import format_date_time

import websocket
from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
    SystemMessage,
)
from langchain.utils import get_from_dict_or_env
from pydantic import Extra, Field, root_validator
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    import tiktoken

logger = logging.getLogger(__name__)


def _import_tiktoken() -> Any:
    try:
        import tiktoken
    except ImportError:
        raise ValueError(
            "Could not import tiktoken python package. "
            "This is needed in order to calculate get_token_ids. "
            "Please install it with `pip install tiktoken`."
        )
    return tiktoken


def _create_retry_decorator(llm: ChatXfyun) -> Callable[[Any], Any]:
    min_seconds = 1
    max_seconds = 60
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        # retry=(
        #     retry_if_exception_type(openai.error.Timeout)
        #     | retry_if_exception_type(openai.error.APIError)
        #     | retry_if_exception_type(openai.error.APIConnectionError)
        #     | retry_if_exception_type(openai.error.RateLimitError)
        #     | retry_if_exception_type(openai.error.ServiceUnavailableError)
        # ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


async def acompletion_with_retry(llm: ChatXfyun, **kwargs: Any) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        # Use OpenAI's async api https://github.com/openai/openai-python#async-api
        return await llm.client.acreate(**kwargs)

    return await _completion_with_retry(**kwargs)


def _convert_dict_to_message(_dict: dict) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        return AIMessage(content=_dict["content"])
    elif role == "system":
        return SystemMessage(content=_dict["content"])
    else:
        return ChatMessage(content=_dict["content"], role=role)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    else:
        raise ValueError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


class ChatXfyun(BaseChatModel):
    """Wrapper around Xfyun Chat large language models.

    environment variable ``XFYUN_APP_ID`` set with your app id.
    environment variable ``XFYUN_API_KEY`` set with your API key.
    environment variable ``XFYUN_API_SECRET`` set with your API Secret.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatOpenAI
            openai = ChatOpenAI(model_name="gpt-3.5-turbo")
    """

    @property
    def lc_serializable(self) -> bool:
        return True

    client: Any  #: :meta private:
    model_name: str = Field(default="gpt-3.5-turbo", alias="model")
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    xfyun_app_id: Optional[str] = None
    """Holds any model parameters valid for `create` call not explicitly specified."""
    xfyun_api_key: Optional[str] = None
    xfyun_api_secret: Optional[str] = None
    """Base URL path for API requests, 
    leave blank if not using a proxy or service emulator."""
    xfyun_api_base: Optional[str] = None
    xfyun_organization: Optional[str] = None
    # to support explicit proxy for OpenAI
    xfyun_proxy: Optional[str] = None
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for requests to OpenAI completion API. Default is 600 seconds."""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""
    n: int = 1
    """Number of chat completions to generate for each prompt."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.ignore
        allow_population_by_field_name = True

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = cls.all_required_field_names()
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["xfyun_app_id"] = get_from_dict_or_env(
            values, "xfyun_app_id", "XFYUN_APP_ID"
        )
        values["xfyun_api_key"] = get_from_dict_or_env(
            values, "xfyun_api_key", "XFYUN_API_KEY"
        )
        values["xfyun_api_secret"] = get_from_dict_or_env(
            values, "xfyun_api_secret", "XFYUN_API_SECRET"
        )
        values["xfyun_organization"] = get_from_dict_or_env(
            values,
            "xfyun_organization",
            "XFYUN_ORGANIZATION",
            default="",
        )
        values["xfyun_api_base"] = get_from_dict_or_env(
            values,
            "xfyun_api_base",
            "XFYUN_API_BASE",
            default="wss://spark-api.xf-yun.com/v1.1/chat",
        )
        values["xfyun_proxy"] = get_from_dict_or_env(
            values,
            "xfyun_proxy",
            "XFYUN_PROXY",
            default="",
        )

        # try:
        #     values["client"] = None
        # except AttributeError:
        #     raise ValueError(
        #         "`xfyun` client not found "
        #     )
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            "model": self.model_name,
            "request_timeout": self.request_timeout,
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
            "n": self.n,
            "temperature": self.temperature,
            **self.model_kwargs,
        }

    def _create_retry_decorator(self) -> Callable[[Any], Any]:
        min_seconds = 1
        max_seconds = 60
        # Wait 2^x * 1 second between each retry starting with
        # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
        return retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
            # retry=(
            #         retry_if_exception_type(openai.error.Timeout)
            #         | retry_if_exception_type(openai.error.APIError)
            #         | retry_if_exception_type(openai.error.APIConnectionError)
            #         | retry_if_exception_type(openai.error.RateLimitError)
            #         | retry_if_exception_type(openai.error.ServiceUnavailableError)
            # ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )

    def create_url(self):
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        host = urlparse(self.xfyun_api_base).netloc
        path = urlparse(self.xfyun_api_base).path
        gpt_url = self.xfyun_api_base

        # 拼接字符串
        signature_origin = "host: " + host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.xfyun_api_secret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.xfyun_api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": host
        }
        # 拼接鉴权参数，生成url
        url = gpt_url + '?' + urlencode(v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        return url

    content = ""
    ws_status = 1

    def completion_with_retry(self, **kwargs: Any) -> Any:

        """Use tenacity to retry the completion call."""
        retry_decorator = self._create_retry_decorator()

        def on_open(ws):
            thread.start_new_thread(run, (ws,))

        def run(ws, *args):
            data = json.dumps(gen_params(appid=ws.appid, messages=ws.messages))
            ws.send(data)

        def gen_params(appid, messages):
            """
            通过appid和用户的提问来生成请参数
            """
            data = {
                "header": {
                    "app_id": appid,
                    "uid": "1234"
                },
                "parameter": {
                    "chat": {
                        "domain": "general",
                        "random_threshold": 0.5,
                        "max_tokens": 2048,
                        "auditing": "default"
                    }
                },
                "payload": {
                    "message": {
                        "text": messages['messages']
                    }
                }
            }
            return data

        def on_message(ws, message):
            data = json.loads(message)
            code = data['header']['code']
            if code != 0:
                print(f'请求错误: {code}, {data}')
                ws.close()
            else:
                choices = data["payload"]["choices"]
                status = choices["status"]
                self.content += choices["text"][0]["content"]
                if status == 2:
                    print(self.content, end='')
                    self.ws_status = 2
                    ws.close()

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            websocket.enableTrace(False)
            ws_url = self.create_url()
            ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_open=on_open)
            ws.appid = self.xfyun_app_id
            ws.messages = kwargs
            ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

            while self.ws_status == 1:
                continue
            return self.content

        return _completion_with_retry(**kwargs)

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            for k, v in token_usage.items():
                if k in overall_token_usage:
                    overall_token_usage[k] += v
                else:
                    overall_token_usage[k] = v
        return {"token_usage": overall_token_usage, "model_name": self.model_name}

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        if self.streaming:
            inner_completion = ""
            role = "assistant"
            params["stream"] = True
            for stream_resp in self.completion_with_retry(
                    messages=message_dicts, **params
            ):
                role = stream_resp["choices"][0]["delta"].get("role", role)
                token = stream_resp["choices"][0]["delta"].get("content", "")
                inner_completion += token
                if run_manager:
                    run_manager.on_llm_new_token(token)
            message = _convert_dict_to_message(
                {"content": inner_completion, "role": role}
            )
            return ChatResult(generations=[ChatGeneration(message=message)])
        response = self.completion_with_retry(messages=message_dicts, **params)
        return self._create_chat_result(response)

    def _create_message_dicts(
            self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = dict(self._invocation_params)
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            gen = ChatGeneration(message=message)
            generations.append(gen)
        llm_output = {"token_usage": response["usage"], "model_name": self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)

    async def _agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        if self.streaming:
            inner_completion = ""
            role = "assistant"
            params["stream"] = True
            async for stream_resp in await acompletion_with_retry(
                    self, messages=message_dicts, **params
            ):
                role = stream_resp["choices"][0]["delta"].get("role", role)
                token = stream_resp["choices"][0]["delta"].get("content", "")
                inner_completion += token
                if run_manager:
                    await run_manager.on_llm_new_token(token)
            message = _convert_dict_to_message(
                {"content": inner_completion, "role": role}
            )
            return ChatResult(generations=[ChatGeneration(message=message)])
        else:
            response = await acompletion_with_retry(
                self, messages=message_dicts, **params
            )
            return self._create_chat_result(response)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    @property
    def _invocation_params(self) -> Mapping[str, Any]:
        """Get the parameters used to invoke the model."""
        openai_creds: Dict[str, Any] = {
            "app_id": self.xfyun_app_id,
            "api_key": self.xfyun_api_key,
            "api_secret": self.xfyun_api_secret,
            "api_base": self.xfyun_api_base,
            "organization": self.xfyun_organization,
            "model": self.model_name,
        }
        # todo
        # if self.xfyun_proxy:
        #     import openai
        #
        #     openai.proxy = {"http": self.openai_proxy,
        #                     "https": self.openai_proxy}  # type: ignore[assignment]  # noqa: E501
        return {**openai_creds, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "xfyun-chat"

    def _get_encoding_model(self) -> Tuple[str, tiktoken.Encoding]:
        tiktoken_ = _import_tiktoken()
        model = self.model_name
        if model == "gpt-3.5-turbo":
            # gpt-3.5-turbo may change over time.
            # Returning num tokens assuming gpt-3.5-turbo-0301.
            model = "gpt-3.5-turbo-0301"
        elif model == "gpt-4":
            # gpt-4 may change over time.
            # Returning num tokens assuming gpt-4-0314.
            model = "gpt-4-0314"
        # Returns the number of tokens used by a list of messages.
        try:
            encoding = tiktoken_.encoding_for_model(model)
        except KeyError:
            logger.warning("Warning: model not found. Using cl100k_base encoding.")
            model = "cl100k_base"
            encoding = tiktoken_.get_encoding(model)
        return model, encoding

    def get_token_ids(self, text: str) -> List[int]:
        """Get the tokens present in the text with tiktoken package."""
        # tiktoken NOT supported for Python 3.7 or below
        if sys.version_info[1] <= 7:
            return super().get_token_ids(text)
        _, encoding_model = self._get_encoding_model()
        return encoding_model.encode(text)

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Calculate num tokens for gpt-3.5-turbo and gpt-4 with tiktoken package.

        Official documentation: https://github.com/openai/openai-cookbook/blob/
        main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb"""
        if sys.version_info[1] <= 7:
            return super().get_num_tokens_from_messages(messages)
        model, encoding = self._get_encoding_model()
        if model == "gpt-3.5-turbo-0301":
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_message = 4
            # if there's a name, the role is omitted
            tokens_per_name = -1
        elif model == "gpt-4-0314":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"get_num_tokens_from_messages() is not presently implemented "
                f"for model {model}."
                "See https://github.com/openai/openai-python/blob/main/chatml.md for "
                "information on how messages are converted to tokens."
            )
        num_tokens = 0
        messages_dict = [_convert_message_to_dict(m) for m in messages]
        for message in messages_dict:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        # every reply is primed with <im_start>assistant
        num_tokens += 3
        return num_tokens
