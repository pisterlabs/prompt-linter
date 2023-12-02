import openai
import safeGPT
from typing import List

class OpenAIChatCompletionWrapper:
    """This class wraps the OpenAI ChatCompletion class to record request parameters
    so that making retries is easier.

    All handlers optionally edit this class object then make new request
    """
    __slots__ = ("model", "messages", "temperature", "top_p", "n", "stream",
                 "presence_penalty", "frequency_penalty", "logit_bias", "args", "kwargs", "response")

    def __init__(self,
                 model: str,
                 messages: List[dict],
                 temperature: float = 1,
                 top_p: float = 1,
                 n: int = 1,
                 stream: bool = False,
                 presence_penalty: float = 0,
                 frequency_penalty: float = 0,
                 logit_bias: dict = None,
                 *args, **kwargs
                 ):
        if n != 1:
            raise ValueError("safeGPT: n!=1 is not currently supported, "
                             "raise an issue if you need this feature")
        if stream:
            raise ValueError("safeGPT: Stream is not supported")
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stream = stream
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias if logit_bias is not None else {}
        self.args = args
        self.kwargs = kwargs

        self.response = None

    def execute(self):
        """This method calls OpenAI."""
        openai.api_key = safeGPT.api_key
        self.response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            top_p=self.top_p,
            n=self.n,
            stream=self.stream,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            logit_bias=self.logit_bias,
            *self.args, **self.kwargs
        )
        return self