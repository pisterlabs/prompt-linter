from tqdm import tqdm
import numpy as np

import openai
from langchain.embeddings import OpenAIEmbeddings
import tiktoken


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def embed_programs_gpt(args, progs):
    """Embed programs using OpenAI's GPT endpoint"""
    progs = [prog.strip() for prog in progs if prog.strip() != ""]
    embeddings_model = OpenAIEmbeddings(disallowed_special=())

    try:
        prog_embeddings = embeddings_model.embed_documents(progs)
    except openai.error.APIError as e:
        raise RuntimeError(f"OpenAI API error: {e}")

    prog_embeddings = np.array(prog_embeddings).reshape(-1, 1, 1536)

    return prog_embeddings
