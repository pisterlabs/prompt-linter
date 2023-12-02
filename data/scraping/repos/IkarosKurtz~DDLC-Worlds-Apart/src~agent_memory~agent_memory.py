from ..character_data import CharacterDetails
from .memory import MemoryEntry, MemoryKind
from ..custom_logger import CustomLogger
from ..agent_memory_manager import AgentMemoryManager
from ..openai_helpers.chat_completion import chat_completion
from openai.embeddings_utils import get_embedding, cosine_similarity
from ..decision_making.thread_decorator import threaded

import textwrap


class AgentMemory:
  """ Manages the agent's memory stream. """

  def __init__(self, initial_memories: list[str], character_data: CharacterDetails, logger: CustomLogger, memory_db: AgentMemoryManager) -> None:
    """
    Initialize the AgentMemory with initial memories, character data, logger, and memory database manager.

    Parameters
    ----------
    initial_memories : list[str]
        List of initial memories as strings to load into the agent's memory.

    character_data : CharacterDetails
        Character data containing information about the agent.

    logger : CustomLogger
        Logger to log information related to the agent's activities.

    memory_db : AgentMemoryManager
        Database manager for storing and retrieving memories.
    """
    self._character_data = character_data
    self._logger = logger
    self._memory_db = memory_db

    self._all_memories: list[MemoryEntry] = []
    self._is_initial_run: bool = True

    self._logger.agent_info("Initializing memories")

    _ = [self._load_initial_memories(initial_memories[i: i + 5]) for i in range(0, len(initial_memories), 5)]

    for stored_memory in self._memory_db.retrieve_all_memories():
      self._all_memories.append(MemoryEntry(**stored_memory))

  @threaded
  def _load_initial_memories(self, memories) -> None:
    """
    Loads initial memories into the agent's memory stream.

    Parameters
    ----------
    memories : list
        A sublist of initial memories to load.
    """
    for memory in memories:
      if self._memory_db.retrieve_memory(memory) is None:
        self._is_initial_run = True
        self.record_memory(memory)
        self._logger.memory_info(f"Stored memory: {memory}")

      self._is_initial_run = False
      self._logger.memory_info(f"Memory: {memory} already exists in the database")

  @property
  def memories(self) -> list[MemoryEntry]:
    """
    Retrieves all memories sorted by most recent.

    Returns
    -------
    list of MemoryEntry
        A sorted list of memory entries.
    """
    self._all_memories.sort(key=lambda memory: memory.created_at.timestamp(), reverse=True)

    return self._all_memories

  def record_memory(self, description: str, memory_kind: MemoryKind = MemoryKind.OBSERVATION, associated_memories: list[str] = None) -> None:
    """
    Records a memory in the agent's memory stream.

    Parameters
    ----------
    description : str
        Description of the memory.

    memory_kind : MemoryKind, optional
        The kind of memory, default is MemoryKind.OBSERVATION.

    associated_memories : list[str], optional
        List of associated memories.
    """
    if associated_memories is None:
      associated_memories = []

    prompt = textwrap.dedent("""
    On a scale from 1 to 10, where 1 is purely mundane (e.g. brushing teeth, making bed, walking the usual route)
    and 10 is impactful (e.g., a breakup, college acceptance), rate the potential significance of the following memory. Only use integers.

    Memory:
    {}

    Format:
    Rating: [<FILL IN>]
    """).format(description.strip())

    importance, _ = chat_completion(prompt, self._character_data.bio)
    importance = importance.split(':')[1].strip()

    self._logger.agent_info(f"Memory > '{description}' > was given a weight of > {importance}")

    importance = float(importance)

    new_memory = MemoryEntry(description, importance, memory_kind, associated_memories=associated_memories)

    self._memory_db.store_memory(new_memory.as_dict())

    self._all_memories.append(new_memory)

  def retrieve(self, query_question: str) -> list[MemoryEntry]:
    """
    Retrieves memories relevant to a given query.

    Parameters
    ----------
    query_question : str
        The query question to retrieve memories for.

    Returns
    -------
    list of MemoryEntry
        A sorted list of relevant memory entries.
    """
    recent_memories = self._all_memories[:70]
    query_embedding = get_embedding(query_question, engine='text-embedding-ada-002')

    for memory in recent_memories:
      recency = memory.calculate_recency()
      importance = memory.importance
      relevance = cosine_similarity(query_embedding, memory.embedding)

      recency_normalized = recency / 1
      importance_normalized = (importance - 1) / 9
      relevance_normalized = relevance / 1

      memory.retrieval_value = (recency_normalized + importance_normalized + relevance_normalized)

    recent_memories.sort(key=lambda memory: memory.retrieval_value, reverse=True)

    return recent_memories
