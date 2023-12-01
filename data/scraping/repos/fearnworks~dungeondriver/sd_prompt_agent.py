from langchain.prompts import ChatPromptTemplate
from ai_driver.image_generation.prompt_generation.prompts import (
    SD_PROMPT_RATING_TEMPLATE,
)
from ai_driver.cloud_llm.cloud_chat_agent import (
    CloudChatAgent,
    CloudChatGenerationConfig,
)
from ai_driver.image_generation.prompt_generation.prompt_rating import SDPromptRating
from dataclasses import dataclass
from typing import Any
from loguru import logger
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from ai_driver.langsmith_config import get_client


@dataclass
class SDAgentConfig(CloudChatGenerationConfig):
    temperature: float = 0.0
    sd_prompt_rating_template: str = SD_PROMPT_RATING_TEMPLATE


@dataclass
class SDPromptEvaluation:
    temperature: float
    generated_prompt: str
    evaluation: Any


class CloudSDAgent:
    def __init__(self, config: CloudChatGenerationConfig = SDAgentConfig()):
        self.llm: ChatOpenAI = ChatOpenAI(
            client=get_client(),
            temperature=config.temperature,
            model=config.model,
        )
        self.rate_template = ChatPromptTemplate.from_template(
            config.sd_prompt_rating_template
        )
        self.chat = self.llm  # backwards compat
        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=self.llm, memory=self.memory, verbose=config.verbose
        )
        self.temperature = config.temperature
        self.model = config.model
        self.config = config

    def get_completion(self, prompt):
        logger.info(f"Prompt: {prompt}")
        response = self.chat.predict(prompt)
        logger.info(f"Response: {response}")
        return {"query": prompt, "result": response}

    def rate(self, generated_prompt):
        logger.info(f"Prompt: {generated_prompt}")
        rate_request = self.rate_template.format_messages(prompt=generated_prompt)
        response = self.chat(rate_request)
        return SDPromptRating.parse(response.content)


def get_default_sd_agent() -> CloudSDAgent:
    default_config = SDAgentConfig()
    return CloudSDAgent(default_config)
