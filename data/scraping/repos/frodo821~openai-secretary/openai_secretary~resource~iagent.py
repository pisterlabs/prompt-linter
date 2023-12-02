from typing import Literal, Required, TypedDict

from openai_secretary.resource.emotion import Emotion

RoleType = Literal['system', 'assistant', 'user']


class ContextItem(TypedDict):
  role: Required[RoleType]
  content: Required[str]


class IAgent:
  context: list[ContextItem]
  emotion: Emotion
