from faiss import Index

from .engine import Engine
from ..embeddings import Embedding
from ..contracts import StoreCoreConfig, EmbeddingConfig, EngineType, LoggingMessageTemplate
from ..logging.utils import LoggingUtils


class EngineFactory:

    @staticmethod
    def get_engine(config: StoreCoreConfig, index: Index, embedding: Embedding) -> Engine:

        engine : Engine = None

        if config.engine_type == EngineType.LANGCHAIN:
            from .langchain_engine import LangChainEngine
            engine = LangChainEngine(index, embedding)
        else:
            raise NotImplementedError("This has not been implemented yet.")

        LoggingUtils.sdk_logger(__package__, config).info(
            LoggingMessageTemplate.COMPONENT_INITIALIZED.format(
                component_name=Engine.__name__,
                instance_type=engine.__class__.__name__
            )
        )

        return engine

    @staticmethod
    def get_index_file_relative_path(config: EmbeddingConfig) -> str:
        if config.engine_type == EngineType.LANGCHAIN:
            from .langchain_engine import LangChainEngine
            return LangChainEngine.get_index_file_relative_path()
        else:
            raise NotImplementedError("This has not been implemented yet.")

    @staticmethod
    def get_data_file_relative_path(config: EmbeddingConfig) -> str:
        if config.engine_type == EngineType.LANGCHAIN:
            from .langchain_engine import LangChainEngine
            return LangChainEngine.get_data_file_relative_path()
        else:
            raise NotImplementedError("This has not been implemented yet.")
