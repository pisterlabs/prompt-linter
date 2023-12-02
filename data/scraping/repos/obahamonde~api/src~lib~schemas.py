from typing import List, Literal, Optional

from openai.types.beta.assistant import Assistant
from pydantic import BaseModel, Field  # pylint: disable=E0611

Purpose = Literal["fine-tuning", "assistants"]
Role = Literal["user", "assistant", "system", "function"]
EventType = Literal["message", "done", "call", "run", "assistant"]


class IMessage(BaseModel):
    role: Role = Field(default="user")
    content: str
    file_ids: List[str] = Field(default_factory=list)


class IAssistant(Assistant):
    picture: Optional[str] = None
