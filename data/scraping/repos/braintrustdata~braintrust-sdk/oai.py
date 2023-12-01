import time

from .logger import current_span


class NamedWrapper:
    def __init__(self, wrapped):
        self.__wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self.__wrapped, name)


class ChatCompletionWrapper:
    def __init__(self, create_fn, acreate_fn):
        self.create_fn = create_fn
        self.acreate_fn = acreate_fn

    def create(self, *args, **kwargs):
        params = self._parse_params(kwargs)
        stream = kwargs.get("stream", False)

        span = current_span().start_span(name="OpenAI Chat Completion", **params)
        should_end = True
        try:
            start = time.time()
            raw_response = self.create_fn(*args, **kwargs)
            if stream:

                def gen():
                    try:
                        first = True
                        all_results = []
                        for item in raw_response:
                            if first:
                                span.log(
                                    metrics={
                                        "time_to_first_token": time.time() - start,
                                    }
                                )
                                first = False
                            all_results.append(item if isinstance(item, dict) else item.dict())
                            yield item
                        span.log(output=all_results)
                    finally:
                        span.end()

                should_end = False
                return gen()
            else:
                log_response = raw_response if isinstance(raw_response, dict) else raw_response.dict()
                span.log(
                    metrics={
                        "tokens": log_response["usage"]["total_tokens"],
                        "prompt_tokens": log_response["usage"]["prompt_tokens"],
                        "completion_tokens": log_response["usage"]["completion_tokens"],
                    },
                    output=log_response["choices"],
                )
                return raw_response
        finally:
            if should_end:
                span.end()

    async def acreate(self, *args, **kwargs):
        params = self._parse_params(kwargs)
        stream = kwargs.get("stream", False)

        span = current_span().start_span(name="OpenAI Chat Completion", **params)
        should_end = True
        try:
            start = time.time()
            raw_response = await self.acreate_fn(*args, **kwargs)
            if stream:

                async def gen():
                    try:
                        first = True
                        all_results = []
                        async for item in raw_response:
                            if first:
                                span.log(
                                    metrics={
                                        "time_to_first_token": time.time() - start,
                                    }
                                )
                                first = False
                            all_results.append(item if isinstance(item, dict) else item.dict())
                            yield item
                        span.log(output=all_results)
                    finally:
                        span.end()

                should_end = False
                return gen()
            else:
                log_response = raw_response if isinstance(raw_response, dict) else raw_response.dict()
                span.log(
                    metrics={
                        "tokens": log_response["usage"]["total_tokens"],
                        "prompt_tokens": log_response["usage"]["prompt_tokens"],
                        "completion_tokens": log_response["usage"]["completion_tokens"],
                    },
                    output=log_response["choices"],
                )
                return raw_response
        finally:
            if should_end:
                span.end()

    @classmethod
    def _parse_params(cls, params):
        params = {**params}
        messages = params.pop("messages", None)
        return {
            "input": messages,
            "metadata": params,
        }


class ChatCompletionV0Wrapper(NamedWrapper):
    def __init__(self, chat):
        self.__chat = chat
        super().__init__(chat)

    def create(self, *args, **kwargs):
        return ChatCompletionWrapper(self.__chat.create, self.__chat.acreate).create(*args, **kwargs)

    async def acreate(self, *args, **kwargs):
        return await ChatCompletionWrapper(self.__chat.create, self.__chat.acreate).acreate(*args, **kwargs)


# This wraps 0.*.* versions of the openai module, eg https://github.com/openai/openai-python/tree/v0.28.1
class OpenAIV0Wrapper(NamedWrapper):
    def __init__(self, openai):
        super().__init__(openai)
        self.ChatCompletion = ChatCompletionV0Wrapper(openai.ChatCompletion)


class CompletionsV1Wrapper(NamedWrapper):
    def __init__(self, completions):
        self.__completions = completions
        super().__init__(completions)

    def create(self, *args, **kwargs):
        return ChatCompletionWrapper(self.__completions.create, None).create(*args, **kwargs)


class AsyncCompletionsV1Wrapper(NamedWrapper):
    def __init__(self, completions):
        self.__completions = completions
        super().__init__(completions)

    async def create(self, *args, **kwargs):
        return await ChatCompletionWrapper(None, self.__completions.create).acreate(*args, **kwargs)


class ChatV1Wrapper(NamedWrapper):
    def __init__(self, chat):
        super().__init__(chat)

        import openai

        if type(chat.completions) == openai.resources.chat.completions.AsyncCompletions:
            self.completions = AsyncCompletionsV1Wrapper(chat.completions)
        else:
            self.completions = CompletionsV1Wrapper(chat.completions)


# This wraps 1.*.* versions of the openai module, eg https://github.com/openai/openai-python/tree/v1.1.0
class OpenAIV1Wrapper(NamedWrapper):
    def __init__(self, openai):
        super().__init__(openai)
        self.chat = ChatV1Wrapper(openai.chat)


def wrap_openai(openai):
    """
    Wrap the openai module (pre v1) or OpenAI instance (post v1) to add tracing.
    If Braintrust is not configured, this is a no-op.

    :param openai: The openai module or OpenAI object
    """
    if hasattr(openai, "chat") and hasattr(openai.chat, "completions"):
        return OpenAIV1Wrapper(openai)
    else:
        return OpenAIV0Wrapper(openai)
