"""
Utils for tokenizing
Ref: https://github.com/pinecone-io/examples/blob/master/generation/langchain/handbook/xx-langchain-chunking.ipynb

Note that for the tokenizer we defined the encoder as "cl100k_base". This is a specific
tiktoken encoder which is used by gpt-3.5-turbo.
"""  # noqa: E501
import tiktoken
from functools import partial
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Any, Dict, List

TOKENIZER = "cl100k_base"

# number of tokens per chunk. 4096 gpt3.5-turbo max / 2 / 5
# Divide by 2 to split generation from context. Divide by 5 to account for 5 results
# per query.
CHUNK_SIZE = 375
CHUNK_OVERLAP = 20  # number of tokens overlap between chunks to keep coherence


def split_by_tokenization(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    tokenizer: str = TOKENIZER,
) -> List[Dict[str, Any]]:
    """
    Split text into chunks using tiktoken tokenizer.

    Args:
        text: text to tokenize
        chunk_size: number of tokens per chunk
        chunk_overlap: number of tokens overlap between chunks
        tokenizer: Tiktoken tokenizer to use. Defaults to TOKENIZER.

    Returns:
        chunks: list of text chunks where each chunk is a dict with keys:
            "text": text chunk
            "n_tokens": number of tokens in the text chunk

    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=partial(tiktoken_len, tokenizer=tokenizer),
        separators=["\n\n", "\n", " ", ""],
    )
    text_chunks = text_splitter.split_text(text)

    chunks = []
    for text_chunk in text_chunks:
        n_tokens = tiktoken_len(text_chunk)
        chunk = {"text": text_chunk, "n_tokens": n_tokens}
        chunks.append(chunk)

    return chunks


def tiktoken_len(text: str, tokenizer: str = TOKENIZER) -> List[str]:
    """
    Tokenize a string using tiktoken and then return the
    number of tokens

    Args:
        text: text to tokenize
        tokenizer: Tiktoken tokenizer to use. Defaults to TOKENIZER.

    Returns:
        n_tokens: number of tokens
    """
    tokenizer = tiktoken.get_encoding(TOKENIZER)
    tokens = tokenizer.encode(text, disallowed_special=())
    n_tokens = len(tokens)
    return n_tokens
