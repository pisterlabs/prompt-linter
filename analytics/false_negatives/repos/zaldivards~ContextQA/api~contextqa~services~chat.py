from langchain.agents import initialize_agent, AgentType, Agent
from langchain.chat_models import ChatOpenAI
from langchain import ConversationChain
from langchain.chains.conversation.prompt import DEFAULT_TEMPLATE
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from contextqa import models, settings
from contextqa.utils import memory, prompts
from contextqa.agents.tools import searcher


_MESSAGES = [
    SystemMessagePromptTemplate.from_template(
        """You are a helpful assistant called ContextQA that answer user inputs. You emphasize your name in every greeting.

    
    
    Example: Hello, I am ContextQA, how can I help you?
    """
    ),
    HumanMessagePromptTemplate.from_template("Hi"),
    AIMessagePromptTemplate.from_template("Hello, I am your assitant ContextQA, how may I help you?"),
    SystemMessagePromptTemplate.from_template(DEFAULT_TEMPLATE),
]


def get_llm_assistant(internet_access: bool) -> ConversationChain | Agent:
    """Return certain LLM assistant based on the system configuration

    Parameters
    ----------
    internet_access : bool
        flag indicating whether an assistant with internet access was requested

    Returns
    -------
    ConversationChain | Agent
    """
    llm = ChatOpenAI(temperature=0)

    if internet_access:
        return initialize_agent(
            [searcher],
            llm=llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory.Redis("default", internet_access=True),
            verbose=settings().debug,
            agent_kwargs={
                # "output_parser": CustomOP(),
                # "format_instructions": prompts.CONTEXTQA_AGENT_TEMPLATE,
                "prefix": prompts.PREFIX,
            },
            handle_parsing_errors=True,
        )
    prompt = ChatPromptTemplate.from_messages(_MESSAGES)
    return ConversationChain(llm=llm, prompt=prompt, memory=memory.Redis("default"), verbose=settings().debug)


def qa_service(params: models.LLMQueryRequest) -> models.LLMResult:
    """Chat with the llm

    Parameters
    ----------
    params : models.LLMQueryRequest
        request body parameters

    Returns
    -------
    models.LLMResult
        LLM response
    """
    assistant = get_llm_assistant(params.internet_access)
    return models.LLMResult(response=assistant.run(input=params.message))
