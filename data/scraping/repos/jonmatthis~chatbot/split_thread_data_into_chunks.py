from typing import List, Any, Dict

from langchain import OpenAI


def split_thread_data_into_chunks(messages: List[str],
                                  max_tokens_per_chunk: int = 1000) -> List[Dict[str, Any]]:
    chunk = ""
    chunks = []
    token_count = 0
    for message in messages:
        chunk += message + "\n"
        token_count = OpenAI().get_num_tokens(chunk)
        if token_count > max_tokens_per_chunk * .9:  # avoid spilling over token buffer to avoid warnings
            chunks.append({"text": chunk,
                           "token_count": token_count, })
            chunk = message + "\n"  # overlap chunks by one message

    if chunk != "":
        chunks.append({"text": chunk,
                       "token_count": token_count, })
    return chunks
