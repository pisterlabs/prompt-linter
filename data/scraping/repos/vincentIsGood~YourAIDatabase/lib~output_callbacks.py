from typing import Any, TextIO
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.output import LLMResult

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, outputStream: TextIO):
        super().__init__()
        self.outputStream = outputStream

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.outputStream.write(token)
        self.outputStream.flush()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: 'UUID | None' = None, **kwargs: Any) -> Any:
        pass