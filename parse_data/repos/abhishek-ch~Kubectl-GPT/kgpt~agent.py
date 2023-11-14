import openai
from langchain import OpenAI, PromptTemplate
from langchain.agents import (initialize_agent,load_tools)
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.llms.openai import BaseOpenAI
from kgpt.kubectlprompt import *


def create_prompt_template() -> PromptTemplate:
    # in future I may need to create my own Parser for cleaner output
    _kubectl_template = "\n\n".join([PREFIX, FORMAT_INSTRUCTIONS, SUFFIX])

    return PromptTemplate(
        input_variables=["input", "chat_history"],
        template=_kubectl_template,
    )

def verify_prompt(example_input:str) -> None:
    kubectl_prompt_template = create_prompt_template()
    print(
        kubectl_prompt_template.format(
            input=example_input, chat_history=None
        )
)

def get_model() -> BaseOpenAI:
    if openai.api_type == "azure":
        return ChatOpenAI(engine="gpt-35-turbo", model_name="gpt-3.5-turbo")
    else:
        return OpenAI(temperature=0.5)

def verify_output(example_input:str) -> None:
    kubectl_prompt_template = create_prompt_template()
    print(get_model()(
        kubectl_prompt_template.format(
            input=example_input,chat_history=None
        )
    ))


def execute(command: str) -> str:
    llm = get_model()
    kubectl_prompt_template = create_prompt_template()

    tools = load_tools(["terminal"], llm=llm)
    memory = ConversationBufferMemory(memory_key="chat_history")

    agent_chain = initialize_agent(
    tools, llm, agent="zero-shot-react-description", memory=memory,
    verbose=True)

    # TODO Its a dirty hack, actual solution
    # https://github.com/hwchase17/langchain/issues/1358
    try:
        return agent_chain.run(
            input=kubectl_prompt_template.format(
                input=command, chat_history=None)
        )
    except ValueError as ex:
        response = str(ex)
        if not response.startswith("Could not parse LLM output: `"):
            raise ex
        return response.removeprefix("Could not parse LLM output: `").removesuffix("`")
