from typing import List, Optional
import json

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from bespokebots.services.chains.templates import (
    BESPOKE_BOT_MAIN_TEMPLATE
)

class BespokeBotChatPrompt:
    """Prompt for the BespokeBotCustomChain."""

    @classmethod
    def from_user_request(cls, user_request: str) -> PromptTemplate:
        """Return a prompt from a human message."""
        
        return ChatPromptTemplate.from_messages(
            [SystemMessagePromptTemplate.from_template(BESPOKE_BOT_MAIN_TEMPLATE),
            HumanMessagePromptTemplate.from_template(user_request)]
        )