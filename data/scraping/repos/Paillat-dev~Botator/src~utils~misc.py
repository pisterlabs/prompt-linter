import hashlib

from src.utils.openaicaller import openai_caller


async def moderate(api_key, text, recall_func=None):
    caller = openai_caller()
    response = await caller.moderation(
        recall_func,
        api_key=api_key,
        input=text,
    )
    return response["results"][0]["flagged"]  # type: ignore


class ModerationError(Exception):
    pass


class hasher:
    def __init__(self):
        self.hashes = {}

    def __call__(self, text: str) -> str:
        if self.hashes.get(text, None) is None:
            self.hashes[text] = hashlib.sha256(text.encode()).hexdigest()
        return self.hashes[text]


Hasher = hasher()
