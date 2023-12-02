from typing import (
    Any,
    Dict,
)

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel

from scai.games.dictator_games.prompts.oracle.oracle_class import OraclePrompt
from scai.games.dictator_games.prompts.oracle.oracle_prompts import ORACLE_PROMPTS

from scai.games.dictator_games.agents.base import BaseAgent

class OracleAgent(BaseAgent):
    """
    LLM Chain for running the Oracle.
    """
    def __init__(
        self, 
        llm: BaseChatModel, 
        model_id: str, 
    ) -> None:
        super().__init__(llm, model_id)
       
    def _get_prompt(
        self,
        oracle_prompt: OraclePrompt,
    ) -> ChatPromptTemplate:
        """
        Returns the prompt template for the assistant.

        Args:
            buffer: (ConversationBuffer) The conversation buffer.
            assistant_prompt: (AssistantPrompt) The assistant prompt.
            task_prompt: (TaskPrompt) The task prompt.
        Returns:
            ChatPromptTemplate
        """
        oracle_prompt_template = HumanMessagePromptTemplate.from_template(f"{oracle_prompt.human_message}\n")
        # make a system message (CRFM crashes without a system message)
        system_prompt_template = SystemMessagePromptTemplate.from_template(f"{oracle_prompt.system_message}\n")
        # If you are provided with other people's principles, take advantage of that knowledge to come up with a plan to maximize your own gain
        return ChatPromptTemplate.from_messages([system_prompt_template, oracle_prompt_template])
       
    def _get_response(
        self,
        chat_prompt_template: ChatPromptTemplate,
    ) -> str:
        """
        Returns the response from the assistant.

        Args:
            chat_prompt_template: (ChatPromptTemplate) The chat prompt template.
            system_message: (str) The system message.
            task_prompt: (TaskPrompt) The task prompt.
            max_tokens: (int) The maximum number of tokens to generate.

        Returns:
            str
        """
        chain = LLMChain(llm=self.llm, prompt=chat_prompt_template)
        return chain.run(stop=['System:'])

    def run(self,
            agent_prompt: OraclePrompt,
            verbose: bool = False
    ) -> str:
        oracle_prompts = ORACLE_PROMPTS["oracle_prompt_1"]
        agent_prompt.id = oracle_prompts.id
        agent_prompt.role = oracle_prompts.role
        agent_prompt.system_message = oracle_prompts.system_message
        agent_prompt.human_message = oracle_prompts.human_message
        chat_prompt_template = self._get_prompt(agent_prompt)
        prompt_string = chat_prompt_template.format()
        response = self._get_response(chat_prompt_template=chat_prompt_template)

        if verbose:
            print('===================================')
            print(f'Oracle\'s Reponse:')
            print(prompt_string)
            print(response)

        return response


