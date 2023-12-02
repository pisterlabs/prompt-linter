import abc
import random
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from logging import Logger
from logging import getLogger
from typing import Dict
from typing import List

import openai
from openai.openai_object import OpenAIObject
from sqlalchemy.orm import Session

from ..model.orm import AssistantMessageLog
from ..model.orm import Message
from ..model.orm import Prompt
from ..util import Describable
from ..util import observe

_logger: Logger = getLogger(__name__)


class MessageRepository(Describable):
    @abc.abstractmethod
    def get(self) -> Message:
        raise NotImplementedError()


@dataclass(frozen=True)
class StaticMessageRepository(MessageRepository):
    # just for debug
    m: Message

    def describe(self) -> Dict:
        return asdict(self)

    @observe(logger=_logger)
    def get(self) -> Message:
        return self.m


@dataclass(frozen=True)
class DatabaseMessageRepository(MessageRepository):
    session: Session

    def describe(self) -> Dict:
        return {}

    @observe(logger=_logger)
    def get(self) -> Message:
        messages: List[Message] = self.session.query(Message).filter(Message.is_active == True).all()
        return random.sample(messages, k=1)[0]


@dataclass(frozen=True)
class OpenAIMessageRepository(MessageRepository):
    session: Session
    api_type: str
    api_key: str
    api_base: str
    api_version: str
    deployment_id: str
    model_name: str

    def describe(self) -> Dict:
        return {
            "api_type": self.api_type,
            "api_base": self.api_base,
            "api_version": self.api_version,
            "model_name": self.model_name
        }

    @observe(logger=_logger)
    def get(self) -> Message:
        openai.api_key = self.api_key
        openai.api_type = self.api_type
        openai.api_base = self.api_base
        openai.api_version = self.api_version
        prompts: List[Prompt] = self.session.query(Prompt).filter(Prompt.is_active == True).all()
        prompt: Prompt = random.sample(prompts, k=1)[0]

        today: str = datetime.today().strftime("%Y年%m月%d日")
        prompts = [
            {
                "role": "system",
                "content": prompt.system
            },
            {
                "role": "user",
                "content": f"今日は{today}です。{prompt.user}"
            }
        ]

        response: OpenAIObject = openai.ChatCompletion.create(
            model=self.model_name,
            deployment_id=self.deployment_id,
            messages=prompts
        )
        m: str = response.choices[0].message.content

        log: AssistantMessageLog = AssistantMessageLog(
            prompt_id=prompt.id,
            message=m,
        )
        self.session.begin()
        self.session.add(log)
        self.session.commit()

        return Message(message=m)
