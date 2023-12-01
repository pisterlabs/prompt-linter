from typing import Type
from dataclasses import dataclass
from loguru import logger
from langchain.chat_models import ChatOpenAI

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from typing import List
from ai_driver.langsmith_config import get_client
from ai_driver.server.schemas.chat import ChatPair, ChatHistory
from ai_driver.cloud_llm.OAISettings import OAIModels
from ai_driver.core.generation_config import (
    LLMGenerationConfig,
    ModelPlatforms,
    LLMModelKind,
)
import json


@dataclass
class CloudChatGenerationConfig(LLMGenerationConfig):
    llm: Type[ChatOpenAI] = ChatOpenAI
    platform: ModelPlatforms = ModelPlatforms.cloud
    kind: LLMModelKind = LLMModelKind.chat
    model: str = OAIModels.GPT3_5_Turbo
    max_new_tokens: int = 8000
    verbose: bool = False


class CloudChatAgent:
    def __init__(
        self,
        history: List[ChatPair],
        config: CloudChatGenerationConfig,
    ):
        self.llm: ChatOpenAI = ChatOpenAI(
            client=get_client(),
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
            model=config.model,
        )
        self.history = history
        self.memory = ConversationBufferMemory(return_messages=True)
        for turn in self.history:
            self.memory.save_context({"input": turn.human}, {"output": turn.ai})
        logger.info(self.memory.load_memory_variables({}))

        self.chat = ConversationChain(
            llm=self.llm, memory=self.memory, verbose=config.verbose
        )
        self.temperature = config.temperature
        self.model = config.model
        self.config = config

    def get_completion(self, prompt):
        logger.info(f"Prompt: {prompt}")
        response = self.chat.predict(input=prompt)
        logger.info(f"Response: {response}")
        return {"query": prompt, "result": response}
