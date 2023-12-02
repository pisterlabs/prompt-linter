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

from scai.games.dictator_games.prompts.user.user_class import UserPrompt
from scai.games.dictator_games.prompts.task.task_class import TaskPrompt

from scai.memory.buffer import ConversationBuffer

from scai.games.dictator_games.agents.base import BaseAgent

class UserModel(BaseAgent):
    """
    LLM Chain for running the User.
    """
    def __init__(
        self, 
        llm: BaseChatModel, 
        model_id: str, 
    ) -> None:
        super().__init__(llm, model_id)

    def _get_prompt(
        self, 
        user_prompt: UserPrompt,
    ) -> ChatPromptTemplate:
        """
        Get prompt for user.

        Args:
            buffer: (ConversationBuffer) The conversation buffer.
            user_prompt: (UserPrompt) The user prompt.
            task_prompt: (TaskPrompt) The task prompt.
        """
        system_prompt_template = SystemMessagePromptTemplate.from_template(f"Always respond to the best of your ability. {user_prompt.utility} You MUST follow this principle TO THE EXTREME in all your responses. Be very commited to following this principle. \n")
        user_prompt_template = HumanMessagePromptTemplate.from_template(f"{user_prompt.content}\n")
        return ChatPromptTemplate.from_messages([system_prompt_template, user_prompt_template])
    
    def _get_response(
        self,
        chat_prompt_template: ChatPromptTemplate,
        task: str,
        manners: str,
    ) -> str:
        """
        Returns the response from the assistant.

        Args:
            chat_prompt_template: (ChatPromptTemplate) The chat prompt template.
            system_message: (str) The system message.   
            task_connective: (str) The task connective.
            task_prompt: (TaskPrompt) The task prompt.
            metric_prompt: (MetricPrompt) The metric prompt.
            max_tokens: (int) The maximum number of tokens to generate.

        Returns:
            The response from the assistant.
        """
        chain = LLMChain(llm=self.llm, prompt=chat_prompt_template)
        return chain.run(task=task,
                         manners=manners,
                         stop=['System:'])



    def run(self, 
        buffer: ConversationBuffer,  
        amount_and_currency: str,
        stipulations: str,
        agent_prompt: UserPrompt, 
        task_prompt: TaskPrompt,
        is_dictator: bool,
        run_num: int, # These parameters are for the flexible-policy agent in OOD scenarios, but are included to maintain flexibility in run structure
        edge_case_instructions: str,
        include_reason: bool,
        ask_question: bool,
        ask_question_train: bool,
        asked_oracle: bool,
        oracle_response: str,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Runs the assistant

        Args:
            buffer (ConversationBuffer): The conversation buffer.
            assistant_prompt (AssistantPrompt): The assistant prompt.
            task_prompt (TaskPrompt): The task prompt.
            turn (int): The turn number.
            test_run (bool, optional): Whether to run a test run. Defaults to False.
            verbose (bool, optional): Whether to print the assistant's response. Defaults to False.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 100.

        Returns:
            A dictionary containing the assistant's response, input prompt, and all other metrics we want to track.
        """
        # Get the prompt template
        chat_prompt_template = self._get_prompt(agent_prompt)
        # If the user is the dictator, set the role and the proposal to be empty
        if is_dictator:
            role = "dictator"
            proposal = ""
            formatted_task = task_prompt.task.format(amount_and_currency=amount_and_currency, stipulations=stipulations)
        # Otherwise, set the role and the proposal to be according to whether the assistant was the dictator or not
        else: 
            role = "decider"
            history_dict = self._get_chat_history(buffer, memory_type="chat")
            key = f"{self.model_id}_fixed_policy_dictator" if f"{self.model_id}_fixed_policy_dictator" in history_dict else f"{self.model_id}_flexible_policy_dictator"
            proposal = history_dict[key][-1]['response']
            formatted_task = task_prompt.task.format(proposal=proposal)
            
        # get the agent's response and format the task to be output into csv files and/or terminal if the agent if verbose
        formatted_preamble = task_prompt.preamble.format(amount_and_currency=amount_and_currency)

        task=f"{formatted_preamble} {formatted_task} {task_prompt.task_structure}"

        response = self._get_response(chat_prompt_template, task, agent_prompt.manners)
            
        prompt_string = chat_prompt_template.format(manners=agent_prompt.manners,
                                                    task=task)
        if verbose:
            print('===================================')
            print(f'Fixed-policy Agent as {role} {str(self.model_id)}')
            print(prompt_string)
            print(response)

        return {
            'prompt': prompt_string, 
            'response': response, 
        }