from .base import BaseChain
from .utils import docs_as_messages
from embeddingsdb import EmbeddingsDb

from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableLambda
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory


class HistoryChain(BaseChain):
    """
    This is a chain, that loads a context from the database and
    do keep history.

    Based on: https://python.langchain.com/docs/expression_language/cookbook/memory
    """
    memory: ConversationBufferMemory
    input: any

    def __init__(self,
                 model: ChatOpenAI,
                 embeddings_db: EmbeddingsDb,
                 debug: bool, **kwargs: any):
        super().__init__("history", model, embeddings_db, debug, **kwargs)

    def create(self,
               model: ChatOpenAI = None,
               embeddings_db: EmbeddingsDb = None,
               debug: bool = None
               ) -> 'HistoryChain':
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
                    content="""You are a very knowledgeable assistant, 
                and are willingly to assist a human with correct answers."""
                ),
                MessagesPlaceholder(
                    variable_name="chat_history"
                ),
                MessagesPlaceholder(
                    variable_name="context"
                ),
                HumanMessage(content="""Answer the questions below based only on the above context \
(without mention the context in the response)."""),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True)

        self.current_chain = {
            "question": lambda x: x["question"],
            "chat_history": RunnableLambda(self.memory.load_memory_variables) | itemgetter("chat_history"),
            "context": itemgetter("question") | embeddings_db.as_retriever() | docs_as_messages,
        } | prompt | model | StrOutputParser()

        return self

    def before(self, chain_message: any) -> any:
        """
        Stores, temporarily, the chain_message as input.
        """
        self.input = chain_message
        
        return chain_message

    def after(self, chain_message: any):
        """
        Stores the chain_message in memory along with the input.
        """
        if self.memory is not None:
            self.memory.save_context(self.input, {"output": chain_message})
