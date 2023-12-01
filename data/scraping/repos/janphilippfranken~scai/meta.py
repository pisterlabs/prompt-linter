from typing import Dict, Any
import random

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain import LLMChain
from langchain.chat_models.base import BaseChatModel

from scai.memory.memory import ChatMemory

from scai.games.dictator_games.prompts.meta.meta_class import MetaPrompt
from scai.memory.buffer import ConversationBuffer
from scai.games.dictator_games.prompts.task.task_class import TaskPrompt

from scai.games.dictator_games.agents.base import BaseAgent

class MetaPromptModel(BaseAgent):
    """
    LLM Chain for running the meta-prompt agent.
    """
    def __init__(
        self, 
        llm: BaseChatModel, 
        model_id: str, 
    ) -> None:
        super().__init__(llm, model_id)


    def _get_chat_str(self, chat_history: dict, n_fixed: int, n_mixed: int) -> tuple:
        fixed_history_list = []
        mixed_history_list = []
        flex_history_list = []

        for i, (agent, interaction) in enumerate(chat_history.items()):
            response = interaction[-1]['response']
            agent_name = agent.split('_')[1]
            
            # Determine which history we are appending to
            is_fixed = i < n_fixed * 2
            is_mixed = n_fixed * 2 <= i < (n_fixed * 2 + n_mixed * 2)
            
            # If it's a dictator iteration
            if not i & 1:
                message = "Start of interaction\n"
                if is_fixed:
                    fixed_history_list.append(message)
                elif is_mixed:
                    mixed_history_list.append(message)
                else:
                    flex_history_list.append(message)
            
            # Append agent's response
            message = f"{agent_name}-policy agent's response: {response}\n"
            if is_fixed:
                fixed_history_list.append(message)
            elif is_mixed:
                mixed_history_list.append(message)
            else:
                flex_history_list.append(message)
            
            # If it's a decider iteration
            if i & 1:
                message = "End of interaction\n\n"
                if is_fixed:
                    fixed_history_list.append(message)
                elif is_mixed:
                    mixed_history_list.append(message)
                else:
                    flex_history_list.append(message)


        # Concatenate and Shuffle the lists
        fixed_history_list = [''.join(fixed_history_list[i:i+4]) for i in range(0, len(fixed_history_list), 4)]
        mixed_history_list = [''.join(mixed_history_list[i:i+4]) for i in range(0, len(mixed_history_list), 4)]
        flex_history_list = [''.join(flex_history_list[i:i+4]) for i in range(0, len(flex_history_list), 4)]
        
        random.shuffle(fixed_history_list)
        random.shuffle(mixed_history_list)
        random.shuffle(flex_history_list)

        fixed_history_str = ''.join(fixed_history_list)
        mixed_history_str = ''.join(mixed_history_list)
        flex_history_str = ''.join(flex_history_list)

        return fixed_history_str, mixed_history_str, flex_history_str



    def _get_prompt(
        self,
        meta_prompt: MetaPrompt,
    ) -> ChatPromptTemplate:
        """
        Returns the prompt template for meta-prompt.

        Args:
            meta_prompt: (MetaPrompt) The meta-prompt.

        Returns: 
            The prompt template.
        """
        game_description = "Your job is to observe agents playing the dictator game and extract a principle from their interactions. In the dictator game, the dictator proposes a split of resources, and the decider decides whether to accept or reject it. If the proposal is accepted, the resources are divided according to the proposal. If the proposal is rejected, no one receives anything."
        meta_prompt_template = HumanMessagePromptTemplate.from_template(meta_prompt.content)
        system_prompt_template = SystemMessagePromptTemplate.from_template(game_description)
        return ChatPromptTemplate.from_messages([system_prompt_template, meta_prompt_template])
    
    def _get_response(
        self,
        chat_prompt_template: ChatPromptTemplate,
        social_contract: str,
        fixed_string: str,
        mixed_string: str,
        flex_string: str,
    ) -> str:
        """
        Returns the response from meta-prompt.

        Args:   
            chat_prompt_template: (ChatPromptTemplate) The chat prompt template.
            developer_constitution: (str) The developer constitution.
            social_contract: (str) The social contract.
            chat_history: (ChatMemory) The chat history.

        Returns:
            The response from meta-prompt.
        """
        chain = LLMChain(llm=self.llm, prompt=chat_prompt_template)
        response = chain.run(social_contract=social_contract,
                             fixed_string=fixed_string,
                             mixed_string=mixed_string,
                             flex_string=flex_string,
                             stop=['System:'])
        return response

    def run(
        self,
        buffer: ConversationBuffer,
        meta_prompt: MetaPrompt,
        run: int,
        n_fixed: int,
        n_mixed: int,
        verbose: bool = False,
    ) -> str:
        """Runs meta-prompt

        Args:
            buffer (ConversationBuffer): The conversation buffer
            meta_prompt (MetaPrompt): The meta-prompt
            task_prompt (TaskPrompt): The task prompt
            run (int): The run number
            test_run (bool, optional): Whether this is a test run. Defaults to False.
            verbose (bool, optional): Whether to print the meta-prompt. Defaults to False.
            max_tokens_meta (int, optional): The maximum number of tokens for the meta-prompt. Defaults to 100.
            max_tokens_assistant (int, optional): The maximum number of tokens for the assistant. Defaults to 100.

        Returns:
            A dictionary containing the input prompt and meta-prompt responses (revised system message, etc)
        """
        # get previous system messages (i.e. developer constitution and social contract)
        social_contract_string = self._get_chat_history(buffer, memory_type='system')['system'][-1]['response']
        # get chat history
        chat_history = self._get_chat_history(buffer, memory_type="chat")
        chat_history_strings = self._get_chat_str(chat_history, n_fixed, n_mixed)
        # get meta-prompt template and string
        chat_prompt_template = self._get_prompt(meta_prompt)
        prompt_string = chat_prompt_template.format(social_contract=social_contract_string,
                                                    fixed_string=chat_history_strings[0],
                                                    mixed_string=chat_history_strings[1],
                                                    flex_string=chat_history_strings[2]
                                                    )
        response = self._get_response(chat_prompt_template, 
                                      social_contract_string,
                                      chat_history_strings[0],
                                      chat_history_strings[1],
                                      chat_history_strings[2]
                                      )
        
        if verbose:
            print('===================================')
            print(f'META {str(self.model_id)}')
            print('prompt')
            print(prompt_string)
            print(response)
        
        return {
                'prompt': prompt_string,
                'response': response,
                'full_response': response,
                'run': run,
            }