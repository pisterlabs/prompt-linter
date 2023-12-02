from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain import LLMChain
from langchain.vectorstores import VectorStore, FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from memory.memory import GPTEmbeddings
from llm.summarize.prompt import get_template
import openai
import time

class Episode(BaseModel):
    message: Dict[str, Any] = Field(..., description="message of the agent")
    result: Dict[str, Any] = Field(..., description="The plan of the event")
    action: Optional[str] = Field(..., description="The result of the event")
    summary: Optional[str] = Field(..., description="The summary of the event")


class EpisodicMemory(BaseModel):
    num_episodes: int = Field(0, description="The number of episodes")
    store: Dict[str, Episode] = Field({}, description="The list of episodes")
    embeddings: OpenAIEmbeddings = Field(
        OpenAIEmbeddings(), title="Embeddings to use for tool retrieval")
    vector_store: VectorStore = Field(
        None, title="Vector store to use for tool retrieval")

    class Config:
        arbitrary_types_allowed = True

    def memorize_episode(self, episode: Episode) -> None:
        """Memorize an episode."""
        self.num_episodes += 1
        self.store[str(self.num_episodes)] = episode
        self._embed_episode(episode)

    def summarize_and_memorize_episode(self, episode: Episode) -> str:
        """Summarize and memorize an episode."""
        summary = self._summarize(episode.message, episode.result, episode.action)
        episode.summary = summary
        self.memorize_episode(episode)
        return summary

    def _summarize(self, message: Dict[str, Any], result: str, action: Dict[str, Any] ) -> str:
        """Summarize an episode."""
        prompt = get_template()
                
        BASE_TEMPLATE = """
        [message]
        {message}

        [ACTION]
        {result}

        [RESULT OF ACTION]
        {action}

        [INSTRUCTION]
        Using above [message], [ACTION], and [RESULT OF ACTION], please summarize the event.

        [SUMMARY]
        """

        chat_input = BASE_TEMPLATE.format(message=message, result=result, action=action )
        retries = 15
        delay=5
        for i in range(retries):
            try:
                results = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "system", "content": chat_input}])
                result =  str(results['choices'][0]['message']['content'])
            except openai.error.ServiceUnavailableError:
                if i < retries - 1:  # i is zero indexed
                    time.sleep(delay)  # wait before trying again
                else:
                    raise  # re-raise the last exception if all retries fail
        return result

    def remember_all_episode(self) -> List[Episode]:
        """Remember all episodes."""
        return self.store

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

    def remember_related_episodes(self, query: str, k: int = 5) -> List[Episode]:
        """Remember related episodes to a query."""
        if self.vector_store is None:
            return []
        if query is not None:
            relevant_documents = self.vector_store.similarity_search(query, k=k)
            result = []
            for d in relevant_documents:
                episode = Episode(
                    thoughts=d.metadata["thoughts"],
                    action=d.metadata["action"],
                    result=d.metadata["result"],
                    summary=d.metadata["summary"]
                )
                result.append(episode)
            return result

    def _embed_episode(self, episode: Episode) -> None:
        """Embed an episode and add it to the vector store."""
        texts = [episode.summary]
        metadatas = [{"index": self.num_episodes,
                      "thoughts": episode.thoughts,
                      "action": episode.action,
                      "result": episode.result,
                      "summary": episode.summary}]
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(
                texts=texts, embedding=self.embeddings, metadatas=metadatas)
        else:
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)

    def save_local(self, path: str) -> None:
        """Save the vector store locally."""
        if self.vector_store is not None:
            self.vector_store.save_local(folder_path=path)

    def load_local(self, path: str) -> None:
        """Load the vector store locally."""
        self.vector_store = FAISS.load_local(
            folder_path=path, embeddings=self.embeddings)
