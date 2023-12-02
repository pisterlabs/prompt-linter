from .base import BaseChain
from tools.smhi import ForecastTool
from embeddingsdb import EmbeddingsDb

from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


class HistoryWithToolsChain(BaseChain):
    """
    This is a chain, that loads a context from the database, have tools, and
    keep history.
    """

    def __init__(self,
                 model: ChatOpenAI,
                 embeddings_db: EmbeddingsDb,
                 debug: bool, **kwargs: any):
        super().__init__("history-with-tools", model, embeddings_db, debug, **kwargs)

    def create(self,
               model: ChatOpenAI = None,
               embeddings_db: EmbeddingsDb = None,
               debug: bool = None
               ) -> 'HistoryWithToolsChain':
        """
        Create the chain
        :param model: The model. If omitted, the default model is used.
        :param embeddings_db: The embeddings database. If omitted, the default embeddings database is used.
        :param debug: The debug flag. If omitted, the default debug flag is used.
        :return: The runnable chain
        """
        model = model or self.model
        embeddings_db = embeddings_db or self.embeddings_db
        debug = debug or self.debug

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="You are very powerful assistant, but bad at calculating lengths of words."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt
            | model
            | OpenAIFunctionsAgentOutputParser()
        )

        self.current_chain = initialize_agent(
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            llm=model,
            # return_direct=True?? -> https://github.com/Chainlit/chainlit/issues/377
            tools=[ForecastTool()], verbose=debug, return_direct=True
        )

        return self

    def before(self, chain_message: any) -> any:
        """
        Replaces the key question with input, if question is not present.
        """
        if "input" not in chain_message:
            chain_message["input"] = chain_message["question"]
            del chain_message["question"]

        return chain_message
    
    def get_output(self, chunk: any) -> str:
        """
        Get the output from the chunk.
        """
        return chunk["output"]
