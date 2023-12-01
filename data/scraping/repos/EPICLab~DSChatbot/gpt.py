"""Defines gpt bot"""
from __future__ import annotations
from typing import TYPE_CHECKING

from ..comm.message import MessageContext 

if TYPE_CHECKING:
    from ..comm.chat_instance import ChatInstance


class GPTBot:
    """GPT bot that connects to Open AI"""

    def __init__(self):
        self.prompt = "You are a chatbot that can help programming.\n\nQ: {}\nA:"
        self.api_key = ""
        self.model_config = {}

    @classmethod
    def config(cls):
        """Defines configuration inputs for bot"""
        return {
            "prompt": ('textarea', {"value": "You are a chatbot that can help programming.\n\nQ: {}\nA:", "rows": 6}),
            "model": ('datalist', {"value": "text-davinci-003", "options": [
                'babbage',
                'davinci',
                'gpt-3.5-turbo-0301',
                'text-davinci-003',
                'babbage-code-search-code',
                'text-similarity-babbage-001',
                'text-davinci-001',
                'ada',
                'curie-instruct-beta',
                'babbage-code-search-text',
                'babbage-similarity',
                'code-search-babbage-text-001',
                'text-embedding-ada-002',
                'code-cushman-001',
                'whisper-1',
                'gpt-3.5-turbo',
                'code-search-babbage-code-001',
                'audio-transcribe-deprecated',
                'text-ada-001',
                'text-similarity-ada-001',
                'text-davinci-insert-002',
                'ada-code-search-code',
                'ada-similarity',
                'code-search-ada-text-001',
                'text-search-ada-query-001',
                'text-curie-001',
                'text-davinci-edit-001',
                'davinci-search-document',
                'ada-code-search-text',
                'text-search-ada-doc-001',
                'code-davinci-edit-001',
                'davinci-instruct-beta',
                'text-similarity-curie-001',
                'code-search-ada-code-001',
                'ada-search-query',
                'text-search-davinci-query-001',
                'curie-search-query',
                'davinci-search-query',
                'text-davinci-insert-001',
                'babbage-search-document',
                'ada-search-document',
                'text-search-curie-query-001',
                'text-search-babbage-doc-001',
                'text-davinci-002',
                'curie-search-document',
                'text-search-curie-doc-001',
                'babbage-search-query',
                'text-babbage-001',
                'text-search-davinci-doc-001',
                'code-davinci-002',
                'text-search-babbage-query-001',
                'curie-similarity',
                'curie',
                'text-similarity-davinci-001',
                'davinci-similarity',
                'cushman:2020-05-03',
                'ada:2020-05-03',
                'babbage:2020-05-03',
                'curie:2020-05-03',
                'davinci:2020-05-03',
                'if-davinci-v2',
                'if-curie-v2',
                'if-davinci:3.0.0',
                'davinci-if:3.0.0',
                'davinci-instruct-beta:2.0.0',
                'text-ada:001',
                'text-davinci:001',
                'text-curie:001',
                'text-babbage:001'
            ]}),
            "temperature": ('range', {"value": 1, "min": 0, "max": 1, "step": 0.01}),
            "max_tokens": ('range', {"value": 1024, "min": 1, "max": 4000, "step": 1}),
            "top_p": ('range', {"value": 0.9, "min": 0, "max": 1, "step": 0.01}), 
            "frequency_penalty": ('range', {"value": 0, "min": 0, "max": 2, "step": 0.01}), 
            "presence_penalty": ('range', {"value": 0, "min": 0, "max": 2, "step": 0.01}),
            "best_of": ('range', {"value": 1, "min": 1, "max": 20, "step": 1}),
            "api_key": ("file", {"value": ""}),
        }
    
    def _set_config(self, original, data, key, convert=str):
        """Sets config"""
        self.model_config[key] = convert(data.get(key, original[key]))

    def start(self, instance: ChatInstance, data: dict):
        """Initializes bot"""
        original = self.config()
        self.prompt = data.get("prompt", self.prompt)
        self.api_key = data.get("api_key", "").strip()
        self._set_config(original, data, 'model', str)
        self._set_config(original, data, 'temperature', float)
        self._set_config(original, data, 'max_tokens', int)
        self._set_config(original, data, 'top_p', float)
        self._set_config(original, data, 'frequency_penalty', float)
        self._set_config(original, data, 'presence_penalty', float)
        self._set_config(original, data, 'best_of', int)

        instance.history.append(MessageContext.create_message(
            ("I am a GPT bot"),
            "bot"
        ))
        instance.config["enable_autocomplete"] = False

    def refresh(self, instance: ChatInstance):
        """Refresh chatbot"""
        # pylint: disable=no-self-use
        instance.sync_chat("refresh")

    def process_message(self, context: MessageContext) -> None:
        """Processes user messages"""
        # pylint: disable=unused-argument
        import openai
        openai.api_key = self.api_key
        response = openai.Completion.create(
          prompt=self.prompt.format(context.text),
          **self.model_config
        )
        try:
            if response.get("choices") is None or len(response["choices"]) == 0:
                raise Exception("GPT API returned no choices")
            text = response["choices"][0].get("text")
            if text is None:
                raise Exception("GPT API returned no text")
            context.reply(text.strip())
        except Exception:
            import traceback
            context.reply(traceback.format_exc(), "error")
    
        return self

    def process_autocomplete(self, instance: ChatInstance, request_id: int, query: str):
        """Processes user autocomplete query"""
        # pylint: disable=unused-argument
        # pylint: disable=no-self-use
        instance.send({
            "operation": "autocomplete-response",
            "responseId": request_id,
            "items": [],
        })

    def save(self):
        """Saves bot"""
        return {
            "config": self.model_config,
            "prompt": self.prompt,
            "!form": {
                "api_key": ("file", {"value": ""})
            }
        }

    def load(self, data):
        """Loads bot"""
        if "config" in data:
            self.model_config = {**self.model_config, **data["config"]}
        self.prompt = data.get("prompt", self.prompt)
        if form := data.get("!form", None):
            self.api_key = form.get("api_key", "").strip()
