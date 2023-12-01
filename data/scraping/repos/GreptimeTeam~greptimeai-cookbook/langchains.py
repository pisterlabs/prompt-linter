import logging

from greptimeai.langchain.callback import GreptimeCallbackHandler

from langchain.agents import AgentExecutor, OpenAIFunctionsAgent, tool
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import SystemMessage

logging.basicConfig(level=logging.DEBUG)

# setup LangChain
callbacks = [GreptimeCallbackHandler()]


def llm_chain(streaming: bool = False):
    return LLMChain(
        llm=OpenAI(streaming=streaming),
        prompt=PromptTemplate.from_template("{text}"),
        callbacks=callbacks,
    )


def chat_chain(streaming: bool = False):
    TEMPLATE = "You are a helpful assistant"
    system_message_prompt = SystemMessagePromptTemplate.from_template(TEMPLATE)
    HUMAN_TEMPLATE = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    return LLMChain(
        llm=ChatOpenAI(streaming=streaming),
        prompt=chat_prompt,
        callbacks=callbacks,
    )


@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)


def agent_chain():
    system_message = SystemMessage(
        content="You are very powerful assistant, but bad at calculating lengths of words."
    )

    MEMORY_KEY = "chat_history"
    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=MEMORY_KEY)],
    )
    memory = ConversationBufferMemory(memory_key=MEMORY_KEY, return_messages=True)
    agent = OpenAIFunctionsAgent(
        llm=ChatOpenAI(temperature=0), tools=[get_word_length], prompt=prompt
    )
    return AgentExecutor(
        agent=agent, tools=[get_word_length], memory=memory, verbose=True
    )
