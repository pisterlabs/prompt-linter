"""
<https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py>
"""

from langchain.prompts import PromptTemplate

human_prefix = "USER"
ai_prefix = "ASSISTANT"
human_suffix = None
ai_suffix = "</s>"

template = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

{{history}}
{human_prefix}: {{input}}
{ai_prefix}:"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)
