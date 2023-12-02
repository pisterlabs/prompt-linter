import tiktoken
import asyncio
from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import LatexTextSplitter
from langchain.schema.messages import HumanMessage, SystemMessage


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
    # https://platform.openai.com/docs/models
    'text-davinci-002':4097, # deprecated
    'text-davinci-003':4097, # deprecated. 
    'gpt-3.5-turbo-16k':16385, 
    'gpt-3.5-turbo':4097,
    'gpt-4':8192,
    'gpt-3.5-turbo-instruct':4097, # recommended replacement for davinci
}

g_use_completion_api = set(['gpt-3.5-turbo-16k', 
                            'gpt-3.5-turbo', 
                            'gpt-4', 
                            'gpt-3.5-turbo-instruct'])

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

    assert model in g_use_completion_api
    llm = ChatOpenAI(model_name=model, openai_api_key=api_key, temperature=0)

    acc = []
    for prompt in prompts:
            cor = llm.ainvoke(input=[HumanMessage(content=prompt)])
            acc.append(cor)

    raw_outputs = await asyncio.gather(*acc)

    outputs = list(map(lambda m : m.content.strip(), raw_outputs))
    return outputs