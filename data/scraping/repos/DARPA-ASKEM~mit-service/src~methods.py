import tiktoken
import asyncio
from typing import List
import openai ## already inited
## assumes 

from openai import OpenAIError

import re

# def clean_spaces(text):
#     text1 = re.sub(' +', ' ', text)
#     text2 = re.sub('\n+', '\n', text1)
#     return text2

def strip_latex_preamble(text):
    start = text.find('\\begin{document}')
    end_marker = '\\end{document}'
    end = text.find(end_marker)
    end_len = len(end_marker)
    return text[start:end+end_len]

def split_into_chunks(text_tokens, max_chunk_size_tokens : int):
    """ 
    split tokens into chunks of at most max size tokens
    """
    token_splits = []
    curr_pos = 0
    while curr_pos < len(text_tokens):
        split = text_tokens[curr_pos:curr_pos + max_chunk_size_tokens]
        token_splits.append(split)
        curr_pos += max_chunk_size_tokens

    assert sum(token_splits, []) == text_tokens
    for c in token_splits:
        assert len(c) <= max_chunk_size_tokens
    return token_splits


def create_prompt_tasks(prompt, document, model_name, answer_token_length=256, chunk_token_length=None):
    max_context_length = g_context_lengths[model_name]
    if chunk_token_length is None:
        chunk_token_length = max_context_length - answer_token_length

    tokenizer = tiktoken.encoding_for_model(model_name)
    pre_text, post_text = prompt.split('[TEXT]')
    pre_tok, post_tok = tokenizer.encode_batch([pre_text, post_text])
    available_length = chunk_token_length - len(pre_tok) - len(post_tok) - 2 # space before and after 
    text_tokens = tokenizer.encode(document)

    assert available_length > 0
    chunks = split_into_chunks(text_tokens, max_chunk_size_tokens=available_length)
    text_tasks  = tokenizer.decode_batch([pre_tok + chunk + post_tok for chunk in chunks])
    return text_tasks

g_context_lengths = {
    'text-davinci-002':4097,
    'text-davinci-003':4097,
    'gpt-3.5-turbo-16k':16000, ## may be slightly larger
    'gpt-3.5-turbo':4097,
    'gpt-4':8000,
}


g_use_completion_api = set(['gpt-3.5-turbo-16k','gpt-3.5-turbo', 'gpt-4'])


from langchain.text_splitter import LatexTextSplitter

def split_latex_into_chunks(document : str,  # latex
                                 prompt_template : str, 
                                model_name : str | None, # knowing which tokenizer guarantees we dont exceed context length
                                max_total_size: int | None,  # if not given, use max possible based on model
                                max_answer_size: int = 256,
                                chunk_overlap: int = 0):

        

    if model_name is not None: # if know model, use tokenizer to guarantee lengths
        max_context_length = g_context_lengths[model_name]

        if max_total_size is None:
            max_total_size = max_context_length

        assert max_total_size <= max_context_length

        tokenizer = tiktoken.encoding_for_model(model_name)
        encoded_prompt = tokenizer.encode(prompt_template)

        chunk_token_length = min(max_context_length - max_answer_size, max_total_size)

        max_document_chunk_size = chunk_token_length - len(encoded_prompt)

        document_chunks = LatexTextSplitter.from_tiktoken_encoder(model_name=model_name, 
                                                                chunk_size=max_document_chunk_size, 
                                                                chunk_overlap=chunk_overlap).split_text(document)
    else: ## tokenizer info not given, then best effort based on character count
        assert max_total_size is not None
        document_chunks = LatexTextSplitter(chunk_size=max_total_size, 
                                            chunk_overlap=chunk_overlap).split_text(document)

    
    return document_chunks

async def fork_join_requests(prompts, model : str, api_key : str = None):
    """
    send one request per prompt 
    """
    acc = []

    if api_key is not None:
        openai.api_key = api_key

    for prompt in prompts:

        if model in g_use_completion_api:
            # TODO: chat completions lets one split the prompt.
            cor = openai.ChatCompletion.acreate(model=model, 
                                                messages=[
                                                    #{"role":"system", "content":TODO}
                                                    {"role": "user", "content": prompt},
                                                ],
                                                temperature=0.0)
        else:
            cor = openai.Completion.acreate(model=model, prompt=prompt, 
                                            temperature=0.0, max_tokens=256)
            
        acc.append(asyncio.create_task(cor))

    outputs = []
    for cor in acc:        
        # try: # no point in handling error here, just makes things confusing
        response = await cor
        # except OpenAIError as err:   
        #     return f"OpenAI connection error: {err}", False
        if model in g_use_completion_api:
            result = response.choices[0].message.content.strip()
        else:
            result = response.choices[0].text.strip()
        print('openai result:\t', result)
        outputs.append(result)

    return outputs