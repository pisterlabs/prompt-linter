# REF: https://platform.openai.com/docs/api-reference/completions

from . import base
import openai
import json
import sys
import os

def use_key(key):
    openai.api_key = key
if not openai.api_key:
    use_key(os.getenv('OPENAI_API_KEY',''))


class ChatModel(base.CompletionModel):
    RENAME_KWARGS = {'limit':'max_tokens'}


    def transform_one(self, prompt, **kw) -> dict:
        kwargs = self.get_api_kwargs(kw)
        kwargs['stop'] = kwargs.get('stop') or None # FIX empty value
        kwargs['model'] = self.name
        #
        system = kw.get('system','')
        start = kw.get('start','')
        messages = []
        if system:
              messages += [{'role':'system', 'content':system}]
        messages += [{'role':'user', 'content':prompt+start}]
        kwargs['messages'] = messages
        #kwargs['max_tokens'] = limit # TODO
        functions = kw.get('functions',[]) or kwargs.get('functions',[]) # NEW
        if functions:
            kwargs['functions'] = [get_signature(f) for f in functions]
        #
        kwargs = self.rename_kwargs(kwargs)
        resp = openai.ChatCompletion.create(**kwargs)
        output_text = resp['choices'][0]['message'].get('content','')
        fc = function_call = resp['choices'][0]['message'].get('function_call',{})
        #
        out = {}
        if function_call:
            out['output'] = {'function_name':fc['name'], 'arguments':json.loads(fc['arguments'])}
        else:
            out['output'] = start + output_text # TODO: detect and handle start duplication
        out['usage'] = resp['usage']
        out['kwargs'] = kwargs
        out['resp'] = resp
        return out


class TextModel(base.CompletionModel):
    RENAME_KWARGS = {'limit':'max_tokens'}

    def transform_one(self, prompt, **kw) -> dict:
        kwargs = self.get_api_kwargs(kw)
        kwargs['stop'] = kwargs.get('stop') or None # FIX empty value
        kwargs['model'] = self.name
        #
        system = kw.get('system','')
        start = kw.get('start','')
        full_prompt = prompt if not system else f'{system.rstrip()}\n\n{prompt}'
        full_prompt += start
        kwargs['prompt'] = full_prompt
        #
        kwargs = self.rename_kwargs(kwargs)
        resp = openai.Completion.create(**kwargs)
        output_text = resp['choices'][0]['text']
        #
        out = {}
        out['output'] = start + output_text
        out['usage'] = resp['usage']
        out['kwargs'] = kwargs
        out['resp'] = resp
        return out


class EmbeddingModel(base.EmbeddingModel):
    def transform_one(self, text, **kw):
        return self.embed_batch([text], **kw)[0]

    def embed_batch(self, texts, **kw):
        limit = kw.get('limit')
        kwargs = {'input':texts, 'model':self.name}
        resp = openai.Embedding.create(**kwargs)
        #
        out = []
        for x in resp['data']:
            out.append({
                 'output': x['embedding'][:limit],
            })
        return out


import inspect
def get_signature(fun):
    params = inspect.signature(fun).parameters
    param_dict = {p:{} for p in params}
    return {
        'name':fun.__name__,
        'description':fun.__doc__ or '',
        'parameters':{'type':'object','properties':param_dict}
    }
