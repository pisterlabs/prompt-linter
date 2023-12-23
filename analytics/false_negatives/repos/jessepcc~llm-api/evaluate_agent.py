import os

from langchain.llms import AzureOpenAI
from langchain import PromptTemplate, LLMChain

from langchain.chains import RetrievalQA
from langchain.vectorstores.base import VectorStore
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.tools.base import ToolException

from langchain.utilities import GoogleSerperAPIWrapper
import prompts

from dotenv import load_dotenv
import openai


def evaluate_agent(
    resume_vectorstore: VectorStore,
    job_vectorstore: VectorStore,
    chat_memory: VectorStore,
) -> AgentExecutor:
    load_dotenv()
    openai_api_base = os.environ["OPENAI_API_BASE"]
    azure_development_name = os.environ["AZURE_DEVELOPMENT_NAME"]
    openai_api_key = os.environ["OPENAI_API_KEY"]

    # llm = AzureChatOpenAI(
    #     openai_api_base=openai_api_base,
    #     openai_api_version="2023-03-15-preview",
    #     deployment_name=azure_development_name,
    #     openai_api_key=openai_api_key,
    #     openai_api_type="azure",
    #     temperature=0,
    #     max_tokens=256,
    # )

    llm = AzureOpenAI(
        openai_api_base=openai_api_base,
        deployment_name="text-davinci-003",
        model_name="text-davinci-003",
        openai_api_key=openai_api_key,
        temperature=0,
        max_retries=2,
    )

    # create tools
    RESUME_PROMPT = PromptTemplate(
        template=prompts.RESUME_EVA_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    JOB_PROMPT = PromptTemplate(
        template=prompts.JOB_EVA_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    CHAT_PROMPT = PromptTemplate(
        template=prompts.CHAT_PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    resume_chain_type_kwargs = {"prompt": RESUME_PROMPT}
    job_chain_type_kwargs = {"prompt": JOB_PROMPT}
    interview_type_kwargs = {"prompt": CHAT_PROMPT}

    job_retriever = job_vectorstore.as_retriever()
    resume_retriever = resume_vectorstore.as_retriever()
    interview_retriever = chat_memory.as_retriever()

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

    in_retriever = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=interview_retriever,
        chain_type_kwargs=interview_type_kwargs,
    )

    search = GoogleSerperAPIWrapper()

    def _handle_error(error: ToolException) -> str:
        return (
            "The following errors occurred during tool execution:"
            + error.args[0]
            + "Please try another tool."
        )

    tools = [
        Tool(
            func=search.run,
            name="search_info",
            description=prompts.SEARCH_TOOL_DESCRIPTION,
            coroutine=search.arun,
            handle_tool_error=_handle_error,
        ),
        Tool(
            func=re_retriever.run,
            description=prompts.RESUME_TOOL_DESCRIPTION,
            name="resume_record",
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
        Tool(
            func=in_retriever.run,
            description=prompts.CHAT_TOOL_DESCRIPTION,
            name="interview_record",
            coroutine=in_retriever.arun,
            handle_tool_error=_handle_error,
        ),
    ]

    prefix = """You are an honest, experienced career coach with a strong command in language and good communication strategy. You are coaching a job seeker who just finished a job interview. Given the following Interview Record, 'AI' represents the interviewer and 'Human' represents job seeker. 'AI' and 'Human' are only names for referring the two different parties and do not add context to the conversation. Your answer should be industry specific and specific the job positions. Generic answers will be considered as wrong answers. Your suggestions should be as specific as possible. Answer the following questions as best as you can. Action must not be None. You have access to the following tools and can only access the following tools:"""

    format = """Use the following format:
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, must be one of [{tool_names}], you should either take an Action or know final answer
        Action Input: the input to the action, must not be None
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
    """

    suffix = """Begin!
    
    Remember, You are talking to the job seeker who is the interviewee. Your response should not include the name of the tool you used.

    Question: {input}

    Thought: {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools=tools,
        prefix=prefix,
        suffix=suffix,
        format_instructions=format,
        input_variables=["input", "agent_scratchpad"],
    )
    tool_names = ", ".join([tool.name for tool in tools])
    print(">>>> tool name: ", tool_names)

    # model = AzureChatOpenAI(
    #     openai_api_base=openai_api_base,
    #     openai_api_version="2023-03-15-preview",
    #     deployment_name=azure_development_name,
    #     openai_api_key=openai_api_key,
    #     openai_api_type="azure",
    #     temperature=0,
    #     max_tokens=256,
    # )

    model = AzureOpenAI(
        openai_api_base=openai_api_base,
        model_name="gpt-interview",
        deployment_name=azure_development_name,
        openai_api_key=openai_api_key,
        temperature=0.4,
        max_retries=2,
    )

    llm_chain = LLMChain(llm=model, prompt=prompt)
    # print(llm_chain.prompt)
    tool_names = [tool.name for tool in tools]
    evaluate_agent = ZeroShotAgent(
        llm_chain=llm_chain, allowed_tools=tool_names, verbose=True
    )
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=evaluate_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors="Check your output and make sure it in acceptable format!",
    )

    return agent_chain
