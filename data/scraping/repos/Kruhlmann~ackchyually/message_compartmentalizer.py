import sys

import openai
import tiktoken


class OpenAIMessageCompartmentalizer:
    MAX_GPT_3_5_TURBO_TOKENS = 4096

    def __init__(self, max_token_limit: int) -> None:
        self.max_token_limit = max_token_limit

    def compartmentalize_message(self, message: str) -> list[str]:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        total_tokens = len(encoding.encode(message))
        num_chunks = total_tokens // self.max_token_limit + 1
        chunks = []
        for i in range(num_chunks):
            start_index = i * self.max_token_limit
            end_index = (i + 1) * self.max_token_limit
            chunk = message[start_index:end_index]
            chunks.append(chunk)
            sys.stderr.write(f"Created chunk of size {len(chunk)}\n")
        return chunks
