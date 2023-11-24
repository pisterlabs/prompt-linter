import os

from langchain.chat_models import AzureChatOpenAI
from langchain import PromptTemplate

from langchain.chains import RetrievalQA
from langchain.memory import (
    ConversationBufferMemory,
)
from langchain.vectorstores.base import VectorStore
from langchain.agents import Tool, initialize_agent, AgentExecutor
from langchain.tools.base import ToolException

from langchain.utilities import GoogleSerperAPIWrapper


import prompts

from dotenv import load_dotenv
import openai


def chat_agent(
    resume_vectorstore: VectorStore,
    job_vectorstore: VectorStore,
) -> AgentExecutor:
    load_dotenv()

    openai_api_base = os.environ["OPENAI_API_BASE"]
    azure_development_name = os.environ["AZURE_DEVELOPMENT_NAME"]
    openai_api_key = os.environ["OPENAI_API_KEY"]

    chat_model = AzureChatOpenAI(
        openai_api_base=openai_api_base,
        openai_api_version="2023-03-15-preview",
        deployment_name=azure_development_name,
        openai_api_key=openai_api_key,
        openai_api_type="azure",
        streaming=True,
        temperature=0.2,
    )

    # model using to extract information from vectorstore
    llm = AzureChatOpenAI(
        openai_api_base=openai_api_base,
        openai_api_version="2023-03-15-preview",
        deployment_name=azure_development_name,
        openai_api_key=openai_api_key,
        openai_api_type="azure",
        temperature=0,
        max_tokens=256,
    )

    # create memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # create tools
    RESUME_PROMPT = PromptTemplate(
        template=prompts.RESUME_PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    JOB_PROMPT = PromptTemplate(
        template=prompts.JOB_PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    resume_chain_type_kwargs = {"prompt": RESUME_PROMPT}
    job_chain_type_kwargs = {"prompt": JOB_PROMPT}

    job_retriever = job_vectorstore.as_retriever()
    resume_retriever = resume_vectorstore.as_retriever()

    re_retriever = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=resume_retriever,
        chain_type_kwargs=resume_chain_type_kwargs,
    )

    jd_retriever = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=job_retriever,
        chain_type_kwargs=job_chain_type_kwargs,
    )

    def _handle_error(error: ToolException) -> str:
        return (
            "The following errors occurred during tool execution:"
            + error.args[0]
            + "Please try another tool."
        )

    search = GoogleSerperAPIWrapper()

    tools = [
        Tool(
            name="search_info",
            func=search.run,
            description=prompts.SEARCH_TOOL_DESCRIPTION,
            coroutine=search.arun,
            handle_tool_error=_handle_error,
        ),
        Tool(
            func=re_retriever.run,
            description=prompts.RESUME_TOOL_DESCRIPTION,
            name="resume",
            coroutine=re_retriever.arun,
            handle_tool_error=_handle_error,
        ),
        Tool(
            func=jd_retriever.run,
            description=prompts.JOB_TOOL_DESCRIPTION,
            name="job_duties",
            coroutine=jd_retriever.arun,
            handle_tool_error=_handle_error,
        ),
    ]

    # change to 'generate' to ensure meaningful responses
    conversational_agent = initialize_agent(
        tools=tools,
        llm=chat_model,
        agent="chat-conversational-react-description",
        verbose=True,
        # max_iterations=2,
        # early_stopping_method="generate",
        handle_parsing_errors="Check your output and make sure it in acceptable format!",
        memory=memory,
    )

    prompt = conversational_agent.agent.create_prompt(
        tools=tools, system_message=prompts.SYSTEM_MSG, human_message=prompts.HUMAN_MSG
    )

    conversational_agent.agent.llm_chain.prompt = prompt

    return conversational_agent
