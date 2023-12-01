from langchain.chat_models import ChatAnthropic
from langchain.llms import Anthropic
import os

chat_model = ChatAnthropic(
    temperature=0,
    model="claude-v1.3-100k",
    max_tokens_to_sample=800,
    anthropic_api_key=os.environ.get(
        "ANTHROPIC_API_KEY",
    ),
)
llm_model = Anthropic(
    temperature=0,
    model="claude-v1.3-100k",
    anthropic_api_key=os.environ.get(
        "ANTHROPIC_API_KEY",
    ),
)
