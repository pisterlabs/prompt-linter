from abc import ABC, abstractmethod, abstractproperty

# models
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory

# agents
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# fine-tune models
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.prompts.example_selector import LengthBasedExampleSelector

from .Database_Prep import conversation_chain_using_prepared_chroma_vectorstore

from dotenv import load_dotenv
import os
load_dotenv()

class AgentFactory:
    def __init__(self, tools: list[Tool] = [], openai_model_name='gpt-4', temperature=0.0, memory = None, max_tokens=2000):
        """generates agents for the user to interact with the LLM.

        Args:
            tools (list[Tool], optional): Tools/StructuredTools the agent will have access to. Tools cannot be passed in to an agent after calling `agent` 
            to create an agent with different tooling, change the properties and call `agent` again. Defaults to [].

            openai_model_name (str, optional): the openai chat model to use. Defaults to 'gpt-4'.

            temperature (float, optional): how "creative" or determinative the model is. Defaults to 0.0.

            memory (_type_, optional): type of memory the agent will have. Defaults to ConversationSummaryBufferMemory which holds a summary of
            the conversation up to `max_tokens`.

            max_tokens (int, optional): The maximum number of tokens before the ConversationBufferMemory resets. Defaults to 2000.
        """
        llm = ChatOpenAI(temperature=temperature, model=openai_model_name)
        self.memory = memory if memory is not None else ConversationSummaryBufferMemory(
            llm=llm,
            memory_key="chat_history",
            return_intermediate_steps = True,
            return_messages=True,
            max_token_limit=max_tokens)
        self.llm = llm
        self.tools = tools

    def agent(self, agent_type = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True):
        """generate an agent given self.tools, self.llm, self.memory, and agent_type

        Args:
            agent_type (_type_, optional): The type of agent that will be generated. 
            Defaults to AgentType.OPENAI_MULTI_FUNCTIONS to allow for multi-parameter functions.

            verbose (bool, optional): Will console output be verbose? Defaults to True.

        Returns:
            Agent: A chatbot with access to functions/tools and a memory.
        """
        chat_history = MessagesPlaceholder(variable_name="chat_history")
        agent_chain = initialize_agent(self.tools,
                                       self.llm,
                                       agent=agent_type,
                                       verbose=verbose,
                                       memory=self.memory,
                                       intermediate_steps=True,
                                       agent_kwargs = {
                                           "memory_prompts": [chat_history],
                                           "input_variables": ["input", "agent_scratchpad", "chat_history"]
                                       })
        return agent_chain

# region Unimplemented
class PromptLengthLimiter:
    def __init__(self, max_words: int, examples: list, example_prompt: PromptTemplate):
        self.max_words = max_words
        self.examples = examples
        self.example_prompt = example_prompt

    def length_based_selector(self):
        return LengthBasedExampleSelector(
            self.examples, self.example_prompt, self.max_words
        )
class FewShotPromptHandler(ABC):
    def __init__(
        self,
        example_template: PromptTemplate,
        examples: list[str],
        prefix: str,
        suffix: str,
        limiter: PromptLengthLimiter = None,
    ):
        self.example_template = example_template
        self.examples = examples
        self.prefix = prefix
        self.suffix = suffix
        self.prompt_template = None
        self.limiter = limiter

    @abstractmethod
    def _example_prompt(self, variables_list: list[str]) -> PromptTemplate:
        return PromptTemplate(
            input_variables=variables_list, template=self.example_template
        )

    @abstractmethod
    def few_shot_prompt_template(
        self,
        prompt: str,
        input_variables=["query"],
        variables_list: list[str] = ["query", "answer"],
        example_separator="\n",
        limiter: PromptLengthLimiter = None,
    ) -> PromptTemplate:
        self.prompt_template = FewShotPromptTemplate(
            examples=self.examples,
            example_prompt=limiter.length_based_selector()
            if limiter is not None
            else self._example_prompt(variables_list),
            prefix=self.prefix,
            suffix=self.suffix,
            input_variables=input_variables,
            example_separator=example_separator,
        )
        return self.prompt_template
    

class ExampleFactory(ABC):
    @abstractproperty
    def example_template(self) -> str:
        """The example template to use with few shot templates"""
        return self._example_template

    @example_template.setter
    def example_template(self, value):
        self._example_template = value

    @example_template.deleter
    def example_template(self):
        del self._example_template

    @abstractproperty
    def example_prompt(self) -> PromptTemplate:
        """The example prompt to use with few shot templates"""
        return self._example_prompt

    @example_prompt.setter
    def example_prompt(self, value):
        self._example_prompt = value

    @example_prompt.deleter
    def example_prompt(self):
        del self._example_prompt

    @abstractmethod
    def examples(self) -> list[dict[str, str]]:
        """interaction examples between user and LLM"""
# endregion
