"""Logic for creating a query engine from a config file."""
import os

from llama_index import GPTVectorStoreIndex, ServiceContext
from llama_index.embeddings import OpenAIEmbedding
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.llms import OpenAI
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.prompts.chat_prompts import CHAT_REFINE_PROMPT

from app.repo_query.repo_processing import RepoProcessor
from app.utils import load_config

config = load_config("app/repo_query/config.json")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

PICKLE_DOCS_DIR = os.path.join(os.path.dirname(__file__), "data", "pickled_docs")

# Create necessary directories
os.makedirs(PICKLE_DOCS_DIR, exist_ok=True)


class RepoQueryEngine:
    """Repo query engine."""

    def __init__(self, command_args):
        self.command_args = command_args
        self.repo_processor = RepoProcessor(
            self.command_args, GITHUB_TOKEN, PICKLE_DOCS_DIR
        )
        self.g_docs = self.repo_processor.process_repos(config["REPOS"])
        self.service_context = self.create_service_context()
        self.docs = []
        for repo_docs in self.g_docs.values():
            self.docs.extend(repo_docs)
        self.query_engine = self.create_query_engine(
            self.service_context, self.docs, CHAT_REFINE_PROMPT
        )

    @staticmethod
    def create_service_context():
        """Create a service context."""
        return ServiceContext.from_defaults(
            llm=OpenAI(
                temperature=0,
                model=config["MODEL_NAME"],
                max_tokens=config["NUMBER_OF_OUTPUT_TOKENS"],
            ),
            embed_model=OpenAIEmbedding(),
            node_parser=SimpleNodeParser(
                text_splitter=TokenTextSplitter(
                    separator=" ",
                    chunk_size=config["CHUNK_SIZE_LIMIT"],
                    chunk_overlap=config["CHUNK_OVERLAP"],
                    backup_separators=config["BACKUP_SEPARATORS"],
                )
            ),
            context_window=config["CONTEXT_WINDOW"],
            num_output=config["NUMBER_OF_OUTPUT_TOKENS"],
        )

    def create_query_engine(self, service_context, docs, refine_template):
        """Create a query engine."""
        index = GPTVectorStoreIndex.from_documents(
            documents=docs, service_context=service_context
        )
        return index.as_query_engine(refine_template=refine_template)

    def query_repo(self, query: str):
        """Query a repo."""
        answer = self.query_engine.query(query)
        return answer
