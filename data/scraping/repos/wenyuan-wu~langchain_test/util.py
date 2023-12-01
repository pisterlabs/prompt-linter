from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def chatgpt_wrapper(sys_prompt, text):
    load_dotenv()
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-0613",  # update to 0613 model
        temperature=0,
        streaming=True,
        # callbacks=[StreamingStdOutCallbackHandler()] # not needed for wrapper
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(sys_prompt, validate_template=False)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template, validate_template=False)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    # get a chat completion from the formatted messages
    messages = chat_prompt.format_prompt(text=text).to_messages()
    # print(messages)
    result = llm(messages)
    return result.content


def chatgpt_wrapper_16k(sys_prompt, text):
    load_dotenv()
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k",
        temperature=0,
        streaming=True,
        # callbacks=[StreamingStdOutCallbackHandler()] # not needed for wrapper
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(sys_prompt, validate_template=False)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template, validate_template=False)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    messages = chat_prompt.format_prompt(text=text).to_messages()
    result = llm(messages)
    return result.content


def chatgpt_wrapper_4(sys_prompt, text):
    load_dotenv()
    llm = ChatOpenAI(
        model_name="gpt-4-0613",  # update to 0613 model
        temperature=0,
        streaming=True,
        # callbacks=[StreamingStdOutCallbackHandler()] # not needed for wrapper
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(sys_prompt, validate_template=False)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template, validate_template=False)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    # get a chat completion from the formatted messages
    messages = chat_prompt.format_prompt(text=text).to_messages()
    # print(messages)
    result = llm(messages)
    return result.content


def dummy_chatbot(sys_prompt, text):
    return text[::-1]
