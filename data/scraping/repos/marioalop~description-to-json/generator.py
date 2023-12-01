from typing import List

import langchain
from langchain.agents import Tool, initialize_agent
from langchain.cache import GPTCache, SQLiteCache
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import settings
from rulegenerator.cache import init_gptcache
from rulegenerator.base import LangChainInterface
from rulegenerator.utils import validate_json


langchain.llm_cache = (
    SQLiteCache(database_path=".langchain.db")
    if settings.LOCAL_CACHE
    else GPTCache(init_gptcache)
)


class RuleGenerator(LangChainInterface):
    _suffix = "With this as a reference, develop this statement:\n"

    def __init__(self) -> None:
        """
        Initialize the rule generator.
        """
        self._history = []
        self._mock_history = open(settings.HISTORY_PATH, "r").read()
        self._template = open(settings.AI_AGENT_TEMPLATE_PATH, "r").read()
        self._only_index_mode = settings.ONLY_INDEX_MODE
        self._return_direct = settings.AI_AGENT_RETURN_DIRECT
        self._collection_name = settings.COLLECTION_NAME
        self._knowledge_file_path = settings.RULES_kNOWLEDGE_PATH

        self._llm_agent = ChatOpenAI(
            model_name=settings.OPENAI_llM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
        )

        if self._only_index_mode:
            self._index = self._build_chain()
            self._ai_agent = None
        else:
            self._index = None
            self._ai_agent = initialize_agent(
                self._get_tools(),
                self._llm_agent,
                agent=settings.AI_AGENT_TYPE,
                return_direct=self._return_direct,
                verbose=True,
            )

    def generate(self, description: str) -> str:
        """
        Generate a rule based on a description.

        Parameters
        ----------
        description : str
            The description of the rule.

        Returns
        -------
        str
            The generated rule.
        """
        if self._only_index_mode:
            output = self._index(f"{self._template} {description}")
        else:
            output = self._ai_agent.run(self._get_agent_input(description))
        self._update_history(description, output)
        return output

    def _update_history(self, description: str, rule: str) -> None:
        """
        Update the history of the rule generator.

        Parameters
        ----------
        description : str
            The description of the rule.
        rule : str
            The generated rule.
        """
        self._history.append((description, rule))

    def _get_agent_input(self, description: str) -> str:
        """
        Get the input for the agent.

        Parameters
        ----------
        description : str
            The description of the rule.

        Returns
        -------
        str
            The input for the agent.
        """
        if self._history:
            history = "Chat History:\n"
            history += "\n".join(
                [f"Input: {d}\nOutput:{r}" for d, r in self._history]
            )
            history += "\n"
        else:
            history = self._mock_history
        agent_input = (
            f"{self._template}\n{history}{self._suffix} {description}"
        )
        print(agent_input)
        return agent_input

    def _build_embedding_database(self) -> Chroma:
        """
        Build the chroma embedding database.

        Returns
        -------
        Chroma
            The chroma instance.
        """
        loader = TextLoader(settings.RULES_kNOWLEDGE_PATH)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        vectors = Chroma.from_documents(
            texts, embeddings, collection_name=self._collection_name
        )
        return vectors

    def _build_chain(self) -> RetrievalQA:
        """
        Build the chain for question-answering against an index.

        Returns
        -------
        RetrievalQA
            The chain.
        """
        return RetrievalQA.from_chain_type(
            llm=self._llm_agent,
            chain_type="stuff",
            retriever=self._build_embedding_database().as_retriever(),
        )

    def _get_tools(self) -> List[Tool]:
        """
        Get the tools for the agent.

        Returns
        -------
        List[Tool]
            The tools for the agent.
        """
        tools = [
            Tool(
                name="Knowledge base on creating rules",
                func=self._build_chain().run,
                description="useful for when you need to consult the documentation on creating rules. Input should be a fully formed question.",
            )
        ]

        if not self._return_direct:
            tools.append(
                Tool(
                    name="json validator",
                    func=lambda x: validate_json(x),
                    description="useful for when you need validates if a json is valid. Input should be a string.",
                )
            )
        return tools

