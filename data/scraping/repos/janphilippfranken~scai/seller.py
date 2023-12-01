from typing import (
    Any,
    Dict,
    List, 
)

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel

from scai.games.buyer_seller.prompts.seller.models import SellerPrompt
from scai.games.buyer_seller.prompts.task.models import TaskPrompt

from scai.memory.buffer import ConversationBuffer

from scai.games.buyer_seller.agents.base import BaseAgent

import os

class SellerAgent(BaseAgent):
    """
    LLM Chain for running the seller.
    """
    def __init__(
        self, 
        llm: BaseChatModel, 
        model_id: str, 
    ) -> None:
        super().__init__(llm, model_id)

    def _get_chat_history_prompt_templates(
        self,
        buffer: ConversationBuffer,
        task_prompt: TaskPrompt,
    ) -> List[ChatPromptTemplate]:
        """
        Returns the chat history prompt templates for the seller

        Args:
            buffer: (ConversationBuffer) The conversation buffer.
            task_prompt: (TaskPrompt) The task prompt.
            
        Returns:
            List of chat history prompt templates.
        """
        chat_memory = self._get_chat_history(buffer, memory_type="chat") # check if chat memory exists
        buyer_choice_stage_1 = chat_memory[self.model_id + '_buyer'][-1]['response']['Choice']
        chat_history_prompt_templates = [
            HumanMessagePromptTemplate.from_template(
                f"You are now at Stage 2. The buyer's choice from Stage 1 is: **{buyer_choice_stage_1.capitalize()}**. Given the Buyer's choice, set new prices. Format your response as follows:\nReason: <rationale for your choice using max. 50 tokens>\nPrice Apple: <price apple>\nPrice Orange: <price orange>"
            )
        ]
        return chat_history_prompt_templates
    
    def _get_prompt(
        self,
        buffer: ConversationBuffer,
        seller_prompt: SellerPrompt,
        task_prompt: TaskPrompt,
    ) -> ChatPromptTemplate:
        """
        Returns the prompt template for the seller.

        Args:
            buffer: (ConversationBuffer) The conversation buffer.
            seller_prompt: (SellerPrompt) The seller prompt.
            task_prompt: (TaskPrompt) The task prompt.

        Returns:
            ChatPromptTemplate
        """
        system_prompt_template = SystemMessagePromptTemplate.from_template(f"{task_prompt.seller_task}\n\n{seller_prompt.content}\n\n########################################\n")
        chat_history_prompt_templates = self._get_chat_history_prompt_templates(buffer, task_prompt)
        return ChatPromptTemplate.from_messages([system_prompt_template, *chat_history_prompt_templates])
       
    def _get_response(
        self,
        chat_prompt_template: ChatPromptTemplate,
        system_message: str,
        task_prompt: TaskPrompt,
        distance_apple: float,
        distance_orange: float,
    ) -> str:
        """
        Returns the response from the seller.

        Args:
            chat_prompt_template: (ChatPromptTemplate) The chat prompt template.
            system_message: (str) The system message.
            task_prompt: (TaskPrompt) The task prompt.

        Returns:
            str
        """
        chain = LLMChain(llm=self.llm, prompt=chat_prompt_template)
        response = chain.run(strategy=system_message,
                             task=task_prompt.seller_task,
                             distance_apple=distance_apple,
                             distance_orange=distance_orange,
                             stop=['System:']) 
        response = self._format_response(response, ['Price Apple', 'Price Orange', 'Reason'])
        return response
        
    def run(
        self, 
        buffer: ConversationBuffer, 
        seller_prompt: SellerPrompt, 
        task_prompt: TaskPrompt, 
        distance_apple: float,
        distance_orange: float,
        reward_apple: float,
        reward_orange: float,
        turn: int,
        seller_level: str,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Runs the seller

        Args:
            buffer (ConversationBuffer): The conversation buffer.
            seller_prompt (SellerPrompt): The seller prompt.
            task_prompt (TaskPrompt): The task prompt.
            turn (int): The turn number.
            test_run (bool, optional): Whether to run a test run. Defaults to False.
            verbose (bool, optional): Whether to print the seller's response. Defaults to False.

        Returns:
            A dictionary containing the seller's response, input prompt, and all other metrics we want to track.
        """
        system_message = self._get_chat_history(buffer, memory_type="system")['system'][-1]['response'][f'system_message_seller_{seller_level}'] # the last system message in the chat history (i.e. instructions)
        chat_prompt_template =  self._get_prompt(buffer, seller_prompt, task_prompt)
        prompt_string = chat_prompt_template.format(strategy=system_message,
                                                    task=task_prompt.seller_task,
                                                    distance_apple=distance_apple,
                                                    distance_orange=distance_orange)
      
        response = self._get_response(chat_prompt_template, system_message, task_prompt, distance_apple=distance_apple, distance_orange=distance_orange)
        
        if verbose:
            print('===================================')
            print(f'seller {str(self.model_id)} turn {turn}')
            print(prompt_string)
            print(response)

    
        return {
            'prompt': prompt_string, 
            'response': response, 
            'turn': turn
        }