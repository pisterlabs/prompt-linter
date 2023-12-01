from langchain.prompts.chat import (
        ChatPromptTemplate
    )
from ..models.chat.open_ai import OpenAIChat
from ..prompts.business_model_prompt import BusinessModelPrompt

class BusinessModelChain:
    """Class that manages the business model chain"""

    def __init__(self):
        bm_prompt = BusinessModelPrompt()
        self.open_ai_chat = OpenAIChat()
        self.chat_prompt = ChatPromptTemplate.from_messages(
            [
                bm_prompt.system_message_prompt,
                bm_prompt.human_message_prompt
            ]
        )
        
    def get_business_model(self, business_idea=''):
        messages = get_prompt_messages(business_idea=business_idea)
        response = self.chat(messages)
 
        return response
