from .base import BaseChain
from .utils import docs_as_messages
from embeddingsdb import EmbeddingsDb

from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage, BaseMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.schema.output_parser import StrOutputParser


class NoHistoryChain(BaseChain):
    """
    This is a plain chain, that loads a context from the database and
    but do not keep history.
    """

    def __init__(self,
                 model: ChatOpenAI,
                 embeddings_db: EmbeddingsDb,
                 debug: bool, **kwargs: any):
        super().__init__("no-history", model, embeddings_db, debug, **kwargs)

    def create(self,
               model: ChatOpenAI = None,
               embeddings_db: EmbeddingsDb = None,
               debug: bool = None
               ) -> 'NoHistoryChain':
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
                and are willingly to assist the human with correct answers."""
                ),
                MessagesPlaceholder(
                    variable_name="context"
                ),
                HumanMessage(content="""Answer the questions below based only on the above context \
(without mention the context in the response)."""),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )

        self.current_chain = (
            {
                "context": itemgetter("question") | embeddings_db.as_retriever() | docs_as_messages,
                "question": lambda x: x["question"],
            }
            | prompt
            | model
            | StrOutputParser()
        )

        return self
