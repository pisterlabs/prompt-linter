from typing import Optional

from phi.conversation import Conversation
from phi.llm.openai import OpenAIChat

from llm.settings import llm_settings
from llm.storage import vision_conversation_storage


def get_vision_conversation(
    user_name: Optional[str] = None,
    conversation_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Conversation:
    """Get a vision conversation"""

    return Conversation(
        id=conversation_id,
        user_name=user_name,
        llm=OpenAIChat(
            model=llm_settings.gpt_4_vision,
            max_tokens=llm_settings.default_max_tokens,
            temperature=llm_settings.default_temperature,
        ),
        storage=vision_conversation_storage,
        debug_mode=debug_mode,
        monitoring=True,
        meta_data={"conversation_type": "VISION"},
    )
