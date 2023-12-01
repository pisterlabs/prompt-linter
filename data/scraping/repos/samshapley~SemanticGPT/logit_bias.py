import openai
import tiktoken
from typing import List

class LogitBias:
    def __init__(self, api_key: str, model: str, suppressed_phrases: List[str], bias: int, request_timeout: int):
        self.api_key = api_key
        openai.api_key = self.api_key
        self.model = model
        self.suppressed_phrases = suppressed_phrases
        self.bias = bias
        self.request_timeout = request_timeout
        self.encoding = tiktoken.encoding_for_model(self.model)
        self.logit_bias = self._create_logit_bias()


    def _augment_phrases(self, phrases: List[str]) -> List[str]:
        def _iter():
            for p in phrases:
                yield from (" " + p, p + " ", p.lower(), p.upper(), p.capitalize(), p.title())
 
        return list(set(_iter()))

    def _create_logit_bias(self) -> dict:
        phrases = self._augment_phrases(self.suppressed_phrases)
        tokens = list(set([t for p in phrases for t in self.encoding.encode(p)]))
        return {t: self.bias for t in tokens}

    def generate_response(self, prompt: str, temperature: float, system_message: str = None) -> str:
        chat_messages = [{"role": "user", "content": prompt}]
        if system_message:
            chat_messages.insert(0, {"role": "system", "content": system_message})
        response = openai.ChatCompletion.create(
            model=self.model, 
            messages=chat_messages, 
            logit_bias=self.logit_bias, 
            temperature=temperature, 
            request_timeout=self.request_timeout
        )
        return response.choices[0].message.content
