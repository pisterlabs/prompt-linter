# REF: https://cohere.ai/pricing
# REF: https://dashboard.cohere.ai/api-keys
# REF: https://docs.cohere.ai/reference/generate
# REF: https://docs.cohere.ai/reference/embed
# REF: https://docs.cohere.ai/reference/tokenize

from . import base
import cohere
import sys
import os

def use_key(key):
	cohere.api_key = key
if not getattr(cohere, 'api_key', None):
	use_key(os.getenv('CO_API_KEY',''))


class TextModel(base.CompletionModel):
    RENAME_KWARGS  = {'stop':'stop_sequences', 'limit':'max_tokens'}

    def __init__(self, name, kwargs):
        super().__init__(name, kwargs)
        self.client = cohere.Client(cohere.api_key)

    def transform_one(self, prompt, **kw) -> dict:
        kwargs = self.get_api_kwargs(kw)
        kwargs['stop'] = kwargs.get('stop') or [] # FIX empty value
        kwargs['model'] = self.name
        #
        system = kw.get('system','')
        start = kw.get('start','')
        full_prompt = prompt if not system else f'{system.rstrip()}\n\n{prompt}'
        full_prompt += start
        kwargs['prompt'] = full_prompt
        #
        kwargs = self.rename_kwargs(kwargs)
        resp = self.client.generate(**kwargs)
        output_text = resp[0]
        #
        out = {}
        out['output'] = start + output_text # TODO: detect and handle start duplication
        out['usage'] = {} # TODO
        out['kwargs'] = kwargs
        out['resp'] = resp
        # TODO usage
        # TODO error
        return out


class EmbeddingModel(base.EmbeddingModel):

    def __init__(self, name, kwargs):
        super().__init__(name, kwargs)
        self.client = cohere.Client(cohere.api_key)

    def transform_one(self, text, **kw):
        return self.embed_batch([text], **kw)[0]

    def embed_batch(self, texts, **kw):
        limit = kw.get('limit')
        resp = self.client.embed(texts, model=self.name)
        #
        out = []
        for x in resp.embeddings:
            out.append({'output': x[:limit]})
        return out
