import tiktoken
import os
import json
import hashlib
from pathlib import Path
import guidance


class AutoMarkup:
    message = """
    {{#system~}}
    You are an expert at adding XML markup to text documents.
    {{~/system}}

    {{#user~}}
    {{prompt}}

    The text:
    {{input}}
    {{~/user}}

    {{#assistant~}}
    {{gen 'markup' temperature=0 max_tokens=%d}}
    {{~/assistant}}
    """
    model = "gpt-4"
    max_tokens = 8191

    def __init__(self):
        self.llm = guidance.llms.OpenAI(self.model)
        assert (
            self.llm.api_key is not None
        ), "You must provide an OpenAI API key to use the OpenAI LLM. Either pass it in the constructor, set the OPENAI_API_KEY environment variable, or create the file ~/.openai_api_key with your key in it."

    # note that guidance does caching, so I don't need to
    def automarkup(self, input_text: str, prompt: str) -> str:
        # Perform the automarkup
        enc = tiktoken.encoding_for_model(self.model)
        used_tokens = len(enc.encode(input_text + prompt + self.message))
        available_tokens = self.max_tokens - used_tokens
        endpoint = guidance(self.message % available_tokens, llm=self.llm)  # type: ignore
        out = endpoint(input=input_text, prompt=prompt)
        markup = out["markup"]
        doctype_loc = markup.find("<!DOCTYPE")
        if doctype_loc == -1:
            raise ValueError("No DOCTYPE found")
        else:
            markup = markup[doctype_loc:]

        res = markup.strip().strip("`")

        if not res.startswith("<?xml"):
            res = '<?xml version="1.0" encoding="UTF-8"?>\n' + res

        return res
