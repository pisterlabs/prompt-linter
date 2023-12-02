from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import openai
from . import OPENAI_API_KEY


def i_love_programming():
    chat = ChatOpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )
    ans = chat.predict_messages(
        [
            HumanMessage(
                content="Translate this sentence from English to French. I love programming."
            )
        ]
    )
    return ans


def i_love_programming_template():
    template = "You are a helpful assistant that translates {input_language} to {output_language}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    msg = chat_prompt.format_messages(
        input_language="English", output_language="French", text="I love programming."
    )

    return msg


def i_love_programming_chain():
    chat = ChatOpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )

    template = "You are a helpful assistant that translates {input_language} to {output_language}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    ans = chain.run(
        input_language="English", output_language="French", text="I love programming."
    )

    return ans
