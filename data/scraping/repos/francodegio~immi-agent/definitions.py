import yaml

from typing import List, Dict, Optional, Tuple
from copy import deepcopy

from omegaconf import OmegaConf
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage, format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableBranch,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_experimental.chat_models import Llama2Chat
from pydantic import BaseModel, Field
from operator import itemgetter


######################## GLOBAL CONFIGURATION ########################
CONFIG = OmegaConf.create(
    yaml.load(open("config/model.yaml"), Loader=yaml.FullLoader))


########################### DATA STRUCTURES ###########################
class Input(BaseModel):
    prompt: str
    chat_history: Optional[List[Dict]] = None


class Output(BaseModel):
    answer: str
    source_documents: List[str]


class ChatHistory(BaseModel):
    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat"}}
    )
    question: str


############################ DEFINITIONS ############################
def load_txt(path):
    with open(path, "r") as f:
        return f.read()


class ChatBot:

    def __init__(self, config=CONFIG):
        self.CONFIG = config
        self._create_chain()

    def _load_model(self):
        config = self.CONFIG.llm
        llm = LlamaCpp(**config.runtime_args)
        self.llm = llm
        self.model = Llama2Chat(llm=llm)

    def _load_vectorstore(self):
        config = self.CONFIG.vectorstore
        loader = DirectoryLoader("/data", glob="**/*.md")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(**config.text_splitter)
        all_splits = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(**config.model)
        self.vectorstore = FAISS.from_documents(all_splits, embeddings)
        self.retriever = self.vectorstore.as_retriever()

    def _build_templates(self):
        config = self.CONFIG.chain
        cqp = load_txt(config.prompts.CONDENSE_QUESTION_PROMPT)
        ap = load_txt(config.prompts.ANSWER_PROMPT)
        self.CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(cqp)
        self.DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
            template="{page_content}"
        )
        self.ANSWER_PROMPT = ChatPromptTemplate.from_messages(
            [
                ("system", ap),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{question}"),
            ]
        )

    def _combine_documents(self, docs, document_separator="\n\n"):
        doc_strings = [
            format_document(doc, self.DEFAULT_DOCUMENT_PROMPT) for doc in docs
        ]
        return document_separator.join(doc_strings)

    def _format_chat_history(self, chat_history: List[Tuple[str, str]]) -> List:
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer

    def _create_chain(self):
        config = self.CONFIG.chain.similarity

        if not hasattr(self, "CONDENSE_QUESTION_PROMPT"):
            self._build_templates()
        if not hasattr(self, "llm"):
            self._load_model()
        if not hasattr(self, "vectorstore"):
            self._load_vectorstore()

        _search_query = RunnableBranch(
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                    run_name="HasChatHistoryCheck"
                ),
                RunnablePassthrough.assign(
                    chat_history=lambda x: self._format_chat_history(
                        x["chat_history"])
                )
                | self.CONDENSE_QUESTION_PROMPT
                | self.model
                | StrOutputParser(),
            ),
            RunnableLambda(itemgetter("question")),
        )

        _inputs = RunnableMap(
            {
                "question": lambda x: x["question"],
                "chat_history": lambda x: self._format_chat_history(x["chat_history"]),
                "context": _search_query | self.retriever | self._combine_documents,
            }
        ).with_types(input_type=ChatHistory)

        similarity = RunnableMap(
            {
                "answer": lambda x: x,
                "source_documents": RunnableLambda(
                    lambda x: self.vectorstore.similarity_search_with_relevance_scores(
                        query=x, k=config.k, score_threshold=config.score_threshold
                    )
                ),
            }
        )

        self.chain = (
            _inputs | self.ANSWER_PROMPT | self.model | StrOutputParser() | similarity
        )

    def reply(
            self,
            prompt: str,
            chat_history: Optional[List[Tuple[str]]] = None
        ) -> Output:
        chat_history = [] if chat_history is None else chat_history
        result = self.chain.invoke(
            {"question": prompt, "chat_history": chat_history}
        )
        return result


def format_history(msg_history: list):
    hist = deepcopy(msg_history)
    _ = hist.pop(0)

    chat_history = []
    if len(hist) < 2:
        return chat_history
    
    for i in range(0, len(hist)-1, 2):
        chat_history.append(
            (hist[i]['content'], hist[i+1]['content'])
        )
    return chat_history