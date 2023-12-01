import os

from pydantic import BaseModel, Field
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAIChat
from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
)
from langchain.prompts import StringPromptTemplate
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.chains import LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.agent_toolkits import NLAToolkit
from langchain.tools.plugin import AIPlugin
import re
import plugnplai
from langchain.schema import Document
import requests
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

# use env file

load_dotenv()

llm = OpenAI(temperature=0)


class SearchInput(BaseModel):
    query: str = Field(description="should be a search query")


class QuoteInput(BaseModel):
    question: str = Field(
        description="should be a question answer pair, dilineated with |"
    )


@tool("search", args_schema=SearchInput)
def search_api(query: str) -> str:
    """
    Useful for checking what you know about a topic, and then searching for more information.
    Searches the vector store of the notes for results related to the query.
    The vector store of notes is your knowledge base.
    Returns direct raw text from the notes, that you can work with.
    """
    # Get the documents from the vector store
    db = FAISS.load_local("faiss_index", OpenAIEmbeddings())
    docs = db.similarity_search(query, k=16)
    resources = ""
    for doc in docs:
        resources += (
            "From page "
            + str(doc.metadata["page"])
            + ":"
            + doc.page_content[:800]
            + "\n"
        )

    return resources


@tool("quotes", args_schema=QuoteInput)
def generate_quotes(input: str) -> str:
    """
    Useful for when you have a topic and chunks from a document for which you need quotes.
    The input to this tool should be in this format,
    "the input question you must answer | content of the document"
    For example, "What is urban studies? | Urban studies is many things. Urban studies is the study of cities. Urban studies is a growing field"
    would be the input if you want to generate quotes pertaining to the question of what urban studies is, given a document talking about urban studies.
    """
    splitted_input = input.split("|")
    question = splitted_input[0]
    quotes = ("|").join(splitted_input[1:])
    return quoter(question, quotes)


def quoter(question: str, quotes: str) -> str:
    # Get the documents from the vector store
    model = OpenAIChat(model_name="gpt-3.5-turbo-16k", temperature=0)
    prompt = (
        "Format the chunks from the document into quotes for the following question: "
        + question
        + "\n\n based on these chunks from a document:"
        + quotes
    )
    return model.predict(prompt)


tools = [search_api, generate_quotes]

tool_names = [tool.name for tool in tools]


def get_tools(query):
    return [search_api, generate_quotes]


from typing import Callable


# Set up the base template
template = """You are an urban studies professor at Yale-NUS, teaching a course in Architecture in Society. Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}], and only one that must match exactly
Action Input: the input to the action, that matches the type expected by the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, should be a long answer, think about the format that would best suit your final thought

{chat_history}

Return long answers if you need to, don't feel the need to make your answers short. 
Your priority is to answer the question as best you can, with as much information as possible.
Question: {input}
{agent_scratchpad}"""


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    ############## NEW ######################
    # The list of tools available
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        print(self.template.format(**kwargs))
        return self.template.format(**kwargs)


prompt = CustomPromptTemplate(
    template=template,
    tools_getter=get_tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "chat_history", "intermediate_steps"],
)

print(prompt)

memory = ConversationBufferMemory(memory_key="chat_history")


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            return AgentFinish(return_values={"output": llm_output}, log=llm_output)
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        print(action, action_input)
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


output_parser = CustomOutputParser()

llm = OpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)

llm_chain = LLMChain(llm=llm, prompt=prompt)

# tools = get_tools("How do I answer this question about Urban Studies and Architecture?")


agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory, max_steps=10
)

agent_executor.run(
    "summarize what you know about tropical modernism, and then format them in quotes from the document that respond to the question"
)
#


def agentQuery(input: str):
    return agent_executor.run(input)
