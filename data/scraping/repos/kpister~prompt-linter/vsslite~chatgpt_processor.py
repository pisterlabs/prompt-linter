import os
import json
from logging import getLogger
import traceback
from typing import Iterator

from openai import ChatCompletion
from vsslite import LangChainVSSLiteClient


logger = getLogger(__name__)


class ChatGPTFunctionResponse:
    def __init__(self, content: str, role: str = "function", trailing_content: str = None):
        self.content = content
        self.role = role
        self.trailing_content = trailing_content


class ChatGPTFunctionBase:
    name = None
    description = None
    parameters = {"type": "object", "properties": {}}
    is_always_on = False
    
    def get_spec(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }

    def make_trailing_content(self, data: dict = None) -> str:
        pass

    async def aexecute(self, request_text: str, **kwargs) -> ChatGPTFunctionResponse:
        pass


class VSSQAFunction(ChatGPTFunctionBase):
    def __init__(self, name: str, description: str, parameters: dict = None, is_always_on: bool = False, prompt_template: str = None, vss_url: str = "http://127.0.0.1:8000", namespace: str = "default", answer_lang: str = "English", verbose: bool = False):
        super().__init__()
        self.name = name
        self.description = description
        self.parameters = parameters or {"type": "object", "properties": {}}
        self.is_always_on = is_always_on
        self.prompt_template = prompt_template or """Question: {question_text}
        
Please answer the question based on the following conditions.

## Conditions

* The 'information to be based on' below is OpenAI's terms of service. Please create answer based on this content.
* While multiple pieces of information are provided, you do not need to use all of them. Use one or two that you consider most important.
* When providing your answer, quote and present the part you referred to, which is highly important for the user.
* The format should be as follows:

```
{{Answer}}

Quotation: {{Relevant part of the information to be based on}}
```

## Information to be based on

{search_results_text}

* If the information above doesn't contains the answer, reply that you cannot provide the answer because the necessary information is not found.
* Please respond **in {answer_lang}**, regardless of the language of the reference material.
"""
        self.namespace = namespace
        self.answer_lang = answer_lang
        self.verbose = verbose
        self.vss = LangChainVSSLiteClient(vss_url)

    def make_trailing_content(self, data: dict = None) -> str:
        trailing_content = ""

        for d in data["search_results"]:
            if "image_url" in d["metadata"]:
                trailing_content += f"![image]({d['metadata']['image_url']})\n"

        return trailing_content

    async def aexecute(self, question_text: str, **kwargs) -> ChatGPTFunctionResponse:
        search_results_text = ""
        sr = await self.vss.asearch(question_text, namespace=self.namespace)
        for d in sr:
            search_results_text += d["page_content"] + "\n\n------------\n\n"

        qprompt = self.prompt_template.format(
            question_text=question_text,
            search_results_text=search_results_text,
            answer_lang=self.answer_lang
        )

        trailing_content = self.make_trailing_content({"search_results": sr})

        if self.verbose:
            logger.info(f"Prompt: {qprompt}")

        return ChatGPTFunctionResponse(qprompt, "user", trailing_content=trailing_content)


class ChatCompletionStreamResponse:
    def __init__(self, stream: Iterator[str], function_name: str=None):
        self.stream = stream
        self.function_name = function_name

    @property
    def response_type(self):
        return "function_call" if self.function_name else "content"


class ChatGPTProcessor:
    def __init__(self, *,
        api_key: str = None,
        api_base: str = None,
        api_type: str = None,
        api_version: str = None,
        model: str = "gpt-3.5-turbo-16k-0613",
        engine: str = None,
        temperature: float = 1.0,
        max_tokens: int = 0,
        functions: dict = None,
        system_message_content: str = None
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base
        self.api_type = api_type
        self.api_version = api_version
        self.model = model
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.functions = functions or {}
        self.system_message_content = system_message_content
        self.histories = []
        self.history_count = 20

    async def chat_completion_stream(self, messages, temperature: float = None, call_functions: bool = True):
        params = {
            "api_key": self.api_key,
            "messages": messages,
            "model": self.model,
            "temperature": self.temperature if temperature is None else temperature,
            "stream": True,
        }
        if self.api_type == "azure":
            params["api_base"] = self.api_base
            params["api_type"] = self.api_type
            params["api_version"] = self.api_version
            params["engine"] = self.engine

        if self.max_tokens:
            params["max_tokens"] = self.max_tokens

        if call_functions and self.functions:
            params["functions"] = []
            for _, v in self.functions.items():
                params["functions"].append(v.get_spec())
                if v.is_always_on:
                    params["function_call"] = {"name": v.name}
                    logger.info(f"Function Calling is always on: {v.name}")
                    break

        stream_resp = ChatCompletionStreamResponse(await ChatCompletion.acreate(**params))

        async for chunk in stream_resp.stream:
            if chunk:
                if len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0]["delta"]
                    if delta.get("function_call"):
                        stream_resp.function_name = delta["function_call"]["name"]
                    break
        
        return stream_resp

    async def chat(self, text: str) -> Iterator[str]:
        try:
            messages = []
            if self.system_message_content:
                messages.append({"role": "system", "content": self.system_message_content})
            messages.extend(self.histories[-1 * self.history_count:])
            messages.append({"role": "user", "content": text})

            response_text = ""
            stream_resp = await self.chat_completion_stream(messages)

            async for chunk in stream_resp.stream:
                delta = chunk["choices"][0]["delta"]
                if stream_resp.response_type == "content":
                    content = delta.get("content")
                    if content:
                        response_text += delta["content"]
                        yield content

                elif stream_resp.response_type == "function_call":
                    function_call = delta.get("function_call")
                    if function_call:
                        arguments = function_call["arguments"]
                        response_text += arguments

            if stream_resp.response_type == "function_call":
                self.histories.append(messages[-1])
                self.histories.append({
                    "role": "assistant",
                    "function_call": {
                        "name": stream_resp.function_name,
                        "arguments": response_text
                    },
                    "content": None
                })

                function_resp = await self.functions[stream_resp.function_name].aexecute(text, **json.loads(response_text))

                if function_resp.role == "function":
                    messages.append({"role": "function", "content": json.dumps(function_resp.content), "name": stream_resp.function_name})
                else:
                    messages.append({"role": "user", "content": function_resp.content})

                response_text = ""
                stream_resp = await self.chat_completion_stream(messages, temperature=0, call_functions=False)

                async for chunk in stream_resp.stream:
                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content")
                    if content:
                        response_text += content
                        yield content
                
                if function_resp.trailing_content:
                    yield f"\n\n{function_resp.trailing_content}"

            if response_text:
                self.histories.append(messages[-1])
                self.histories.append({"role": "assistant", "content": response_text})

        except Exception as ex:
            logger.error(f"Error at chat: {str(ex)}\n{traceback.format_exc()}")
            raise ex
