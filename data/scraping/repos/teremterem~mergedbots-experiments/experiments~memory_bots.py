import datetime
import math
from typing import Any, AsyncGenerator

import faiss
from langchain import FAISS, InMemoryDocstore
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.experimental import GenerativeAgentMemory
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document
from mergedbots import MergedBot, MergedMessage

from experiments.common.bot_manager import FAST_GPT_MODEL, bot_manager


class PatchedTimeWeightedVectorStoreRetriever(TimeWeightedVectorStoreRetriever):
    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        current_time = kwargs.pop("current_time", None) or datetime.datetime.now()
        return super().add_documents(documents, current_time=current_time, **kwargs)

    async def aadd_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        current_time = kwargs.pop("current_time", None) or datetime.datetime.now()
        return await super().aadd_documents(documents, current_time=current_time, **kwargs)


def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)


def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    embedding_size = 1536
    # TODO save it to disk after every interaction ?
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query, index, InMemoryDocstore({}), {}, relevance_score_fn=relevance_score_fn
    )
    return PatchedTimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=15)


LLM = PromptLayerChatOpenAI(
    model_name=FAST_GPT_MODEL,  # TODO shouldn't this be SLOW_GPT_MODEL ?
    max_tokens=1500,
    pl_tags=["memory"],
)  # Can be any LLM you want.

# TODO memory instance per user
memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=8,  # we will give this a relatively low number to show how reflection works
)


@bot_manager.create_bot(handle="MemoryBot")
async def memory_bot(bot: MergedBot, message: MergedMessage) -> AsyncGenerator[MergedMessage, None]:
    memory.add_memory(f"{message.sender.name.upper()} SAYS: {message.content}")
    yield await message.service_followup_as_final_response(bot, "`MEMORY UPDATED`")


@bot_manager.create_bot(handle="RecallBot")
async def recall_bot(bot: MergedBot, message: MergedMessage) -> AsyncGenerator[MergedMessage, None]:
    memory_docs = memory.fetch_memories(f"{message.sender.name.upper()} SAYS: {message.content}")
    for doc in memory_docs:
        yield await message.service_followup_as_final_response(bot, f"```\n{doc.page_content}\n```")
