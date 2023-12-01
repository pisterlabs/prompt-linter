from expert_gpts.embeddings.llamaindex import LlamaIndexEmbeddingsHandler
from shared.config import EMBEDDINGS_TYPE
from shared.llm_manager_base import BaseLLMManager
from shared.patterns import Singleton


class EmbeddingsHandlerFactory(metaclass=Singleton):
    def get_chain_embeddings(
        self,
        llm_manager: BaseLLMManager,
        embeddings: EMBEDDINGS_TYPE = None,
        load_docs: bool = False,
        index_name: str = "main_chain_memory",
        index_prefix: str = "main_chain_memory_",
    ) -> LlamaIndexEmbeddingsHandler:
        return LlamaIndexEmbeddingsHandler(
            llm_manager,
            embeddings=embeddings,
            index_name=index_name,
            index_prefix=index_prefix,
            load_docs=load_docs,
        )

    def get_expert_embeddings(
        self,
        llm_manager: BaseLLMManager,
        expert_key: str,
        embeddings: EMBEDDINGS_TYPE = None,
        load_docs: bool = False,
    ) -> LlamaIndexEmbeddingsHandler:
        return LlamaIndexEmbeddingsHandler(
            llm_manager,
            embeddings=embeddings,
            index_name=f"{expert_key}_memory",
            index_prefix=f"{expert_key}_memory_",
            load_docs=load_docs,
        )
