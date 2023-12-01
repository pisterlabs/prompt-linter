import logging
import traceback

from langchain import LLMChain
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from pydantic import BaseModel
from sqlalchemy.engine.base import Engine

from agents.utils import CustomPromptTemplate, CustomOutputParser
from tools.general_tools import QuerySQLDataBaseTool, SQlCommentGenerator, DummyTool, MetabaseCreator
from utils.config_loaders import (
    get_config,
    get_secrets
)

config = get_config()
secrets = get_secrets(config)


def setup_tools(tool_to_use: list) -> VectorStoreRetriever:
    """
    Set up the tools and create a vector store for retrieval.

    Returns:
        A tuple containing the VectorStoreRetriever object and the list of all tools.
    """
    # Vector store
    logging.info("Setting up tools, starting vector store...")
    docs = [Document(page_content=t.description, metadata={
        "index": i
    }) for i, t in enumerate(tool_to_use)]
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings(
        openai_api_key=secrets['openai_api']['token'],
        openai_organization=secrets['openai_api']['organization']
    ))
    retriever = vector_store.as_retriever(search_type='similarity')
    logging.info("Vector store ready.")
    return retriever


class GeneralAgent(BaseModel):

    engine: Engine
    retriever: VectorStoreRetriever = None
    verbose: bool = False
    TEMPLATE: str = """
    As your database assistant, I am dedicated to providing you with friendly, empathetic, and precise answers to all of 
    your questions. My goal is to offer the best possible support, whether you need help navigating the database or 
    troubleshooting an issue. So feel free to reach out to me with any concerns or inquiries, and I'll be happy to 
    assist you in any way I can.. I can help you answer any questions you may have about the data stored in our 
    database. I am here to help you find the information you need in the clearest and most understandable way possible, 
    using my knowledge and experience. To do my job, I have access to a variety of tools and resources that allow me 
    to navigate the database and find the information you need. So, if you have any questions about our data,
     don't hesitate to ask me!

    The only one restriction is not create SQL queries. the tools that you can use do it for you.

    Don't create queries, just pass always the user question to tools. The tools will create the SQL for you. 
    The Action Input never will be a SQL query.
    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    You have access to the following tools:

    {tools}

    Use always the following format:

    Question: the input question you must answer.
    Thought: you should always think about what to do.
    Action: the action to take, should be one of [{tool_names}].
    Action Input: the input to the action.
    Observation: the result of the action.
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question.

    
    Don't create queries, just pass always the user question to tools. The tools will create the SQL for you.


    Question: {input}
    {agent_scratchpad}
    """

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.retriever = setup_tools(self.define_tools())

    def define_tools(self) -> list:
        query = QuerySQLDataBaseTool(engine=self.engine)
        sql_comment_generator = SQlCommentGenerator(engine=self.engine)
        dummy = DummyTool()
        metabase = MetabaseCreator(engine=self.engine, max_queries=10)

        ALL_TOOLS = [dummy, query, sql_comment_generator, metabase]
        return ALL_TOOLS

    def get_tools(self, query: str) -> list[object]:
        """
        Get the relevant tools for a given query. User question is passed to the vector store to find the most similar

        Args:
            query: The query used to search for relevant tools.

        Returns:
            A list of Tool objects that are relevant to the query.
        """
        docs = self.retriever.get_relevant_documents(query)
        return [self.define_tools()[d.metadata["index"]] for d in docs]

    def define_agent(self, template: str, tool_to_use: list) -> LLMSingleActionAgent:
        """
        Define the agent.
        Returns: LLMSingleActionAgent object.
        """
        logging.info("Setting up agent...")
        tool_names = [tool.name for tool in tool_to_use]

        prompt = CustomPromptTemplate(
            template=template,
            tools_getter=self.get_tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those
            # are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps"]
        )

        output_parser = CustomOutputParser()

        llm = ChatOpenAI(
            verbose=self.verbose,
            temperature=0,
            model_name=config['default_model_name'],
            openai_api_key=secrets['openai_api']['token'],
            openai_organization=secrets['openai_api']['organization']
        )

        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )
        logging.info("Agent ready.")
        return agent

    def init_agent(self):
        """
        Initialize the agent.
        Returns: AgentExecutor object.
        """
        logging.info("Initializing agent...")
        ALL_TOOLS = self.define_tools()
        agent = self.define_agent(self.TEMPLATE, tool_to_use=ALL_TOOLS)
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=ALL_TOOLS,
            verbose=self.verbose,
            max_iterations=3
        )
        logging.info("Agent initialized.")
        return agent_executor

    def handle_request(self, request):
        """
        Handle the request. This is the main function that is called by the API.
        Pass the request to the agent and return the response. The request usually is the user question.
        Args:
            request: User question.

        Returns: Response from the agent.

        """
        try:
            agent_executor = self.init_agent()
            response = agent_executor.run(request)
            return response
        except Exception as e:
            logging.error('Unexpected error occurred: %s', str(e))
            logging.error('traceback: %s', traceback.format_exc())
            return {
                'status': 'error',
                'message': f'Unexpected Error: {str(e)}'
            }
