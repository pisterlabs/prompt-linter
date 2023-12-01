import logging
import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain.llms.base import BaseLLM
from langchain import LLMChain
from langchain.vectorstores import DeepLake, FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from llm.summarize.prompt import get_template
from ui.cui import CommandlineUserInterface
from utils.constants import DEFAULT_EMBEDDINGS
from utils.util import atimeit, timeit
import base58


class Episode(BaseModel):
    thoughts: Dict[str, Any] = Field(..., description="thoughts of the agent")
    action: Dict[str, Any] = Field(..., description="action of the agent")
    result: str = Field(..., description="The plan of the event")
    summary: str = Field("", description="summary of the event")
    question: str = Field("", description="question to be answered")
    task: str = Field("", description="task to be completed")

    # create like equals method to compare two episodes
    def __eq__(self, other):
        return (
            self.thoughts == other.thoughts
            and self.action == other.action
            and self.result == other.result
        )

    @staticmethod
    def get_summary_of_episodes(episodes: List["Episode"]) -> str:
        return "\n".join([episode.summary for episode in episodes])


class EpisodicMemory(BaseModel):
    num_episodes: int = Field(0, description="The number of episodes")
    store: Dict[str, Episode] = Field({}, description="The list of episodes")
    llm: BaseLLM = Field(..., description="llm class for the agent")
    embeddings: HuggingFaceEmbeddings = Field(DEFAULT_EMBEDDINGS,
        title="Embeddings to use for tool retrieval",
    )
    vector_store: FAISS = Field(
        None, title="Vector store to use for tool retrieval"
    )

    ui: CommandlineUserInterface | None = Field(None)

    class Config:
        arbitrary_types_allowed = True

    # def __init__(self, question: str, **kwargs):
    #     super().__init__(**kwargs)
    #     filename = base58.b58encode(question.encode()).decode()
    #     if self.vector_store is None:
            # self.vector_store = DeepLake(read_only=True, dataset_path=os.path.join(EPISODIC_MEMORY_DIR, f"{filename}"),
            #                              embedding=self.embeddings)

    def __del__(self):
        del self.embeddings
        del self.vector_store

    async def memorize_episode(self, episode: Episode) -> None:
        """Memorize an episode."""
        self.num_episodes += 1
        self.store[str(self.num_episodes)] = episode
        await self._embed_episode(episode)

    async def summarize_and_memorize_episode(self, episode: Episode) -> str:
        """Summarize and memorize an episode."""
        summary = await self._summarize(
            episode.question, episode.task, episode.thoughts, episode.action, episode.result
        )
        episode.summary = summary
        await self.memorize_episode(episode)
        return summary

    async def _summarize(
        self, question: str, task: str, thoughts: Dict[str, Any], action: Dict[str, Any], result: str
    ) -> str:
        """Summarize an episode."""
        prompt = get_template()
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)
        try:
            result = await llm_chain.apredict(
                question=question, task=task, thoughts=thoughts, action=action, result=result
            )
        except Exception as e:
            raise Exception(f"Error: {e}")
        return result

    def remember_all_episode(self) -> List[Episode]:
        """Remember all episodes."""
        # return list(self.store.values())
        return self.store

    @timeit
    def remember_recent_episodes(self, n: int = 5) -> List[Episode]:
        """Remember recent episodes."""
        if not self.store:  # if empty
            return self.store
        n = min(n, len(self.store))
        return list(self.store.values())[-n:]

    def remember_last_episode(self) -> Episode:
        """Remember last episode."""
        if not self.store:
            return None
        return self.store[-1]
    
    @timeit
    def remember_related_episodes(self, query: str, k: int = 5) -> List[Episode]:
        """Remember related episodes to a query."""
        logging.debug('remember_related_episodes')
        if self.vector_store is None:
            return []
        relevant_documents = self.vector_store.similarity_search(query, k=k)
        result = []
        for d in relevant_documents:
            episode = Episode(
                thoughts=d.metadata["thoughts"],
                action=d.metadata["action"],
                result=d.metadata["result"],
                summary=d.metadata["summary"],
                question=d.metadata["question"],
                task=d.metadata["task"]
            )
            result.append(episode)
        return result
   
    @atimeit
    async def _embed_episode(self, episode: Episode) -> None:
        """Embed an episode and add it to the vector store."""
        print('_embed_episode')
        texts = [episode.summary]
        metadatas = [
            {
                "index": self.num_episodes,
                "thoughts": episode.thoughts,
                "action": episode.action,
                "result": episode.result,
                "summary": episode.summary,
                "question": episode.question,
                "task": episode.task
            }
        ]
        if self.vector_store is None:
            print('build deeplake')
        #     self.vector_store = DeepLake(read_only=False, dataset_path=EPISODIC_MEMORY_DIR,embedding=self.embeddings)
            self.vector_store = FAISS.from_texts(
                texts=texts, embedding=self.embeddings, metadatas=metadatas
            )
        else:
            print('_embed_episode::add_texts')
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)

    # async def save_local(self, path: str) -> None:
    #     """Save the vector store locally."""
    #     # async def _save():
    #     print('save_local_inner')
    #     # self.vector_store.save_local(folder_path=path)
    #     # await asyncio.to_thread(vs.save_local, folder_path=path)
    #     print('post save_local inner')
    #     # await asyncio.create_task(_save())

    # def load_local(self, path: str) -> None:
    #     """Load the vector store locally."""
    #     print('local_load inner')
        # async def _load():
        #     self.vector_store = FAISS.load_local(
        #         folder_path=path, embeddings=self.embeddings
        #     )
        # self.vector_store = DeepLake(read_only=False, dataset_path=path,embedding=self.embeddings)
        # await asyncio.create_task(_load())
        # await asyncio.to_thread(FAISS.load_local, folder_path=path, embeddings=self.embeddings)
            
