# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2023 Jesse Palarus
import json

from langchain.callbacks.base import BaseCallbackHandler

from QAChat.Common.asynchronous_processor import AsynchronousProcessor
from QAChat.Common.deepL_translator import DeepLTranslator


class StreamLLMCallbackHandler(BaseCallbackHandler):
    def __init__(self, translator=None):
        self.text = ""
        self.lang = "EN-US"
        self.asynchronous_processor = AsynchronousProcessor(self.send_response)

        self.translator = DeepLTranslator() if translator is None else translator

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.asynchronous_processor.add(self.text)

    def send_response(self, text):
        if self.lang != "EN-US":
            text = self.translator.translate_to(
                text, self.lang, use_spacy_to_detect_lang_if_needed=False
            ).text
        return json.dumps({"text": text}) + "\n"
