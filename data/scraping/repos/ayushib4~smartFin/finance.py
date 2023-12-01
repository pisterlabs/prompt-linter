import openai
import math
import faiss
from dotenv import load_dotenv
from os import getenv
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS

load_dotenv()

openai.organization = getenv("OPENAI_ORG_ID")
openai.api_key = getenv("OPENAI_API_KEY")

NAME = "Mr. Wonderful"
AGE = 68
TRAITS = """ Kevin have a straightforward and practical approach to personal finance, 
    emphasizing disciplined budgeting and prioritizing financial goals to help everyday people make sound spending decisions. 
    Kevin emphasizes the importance of tracking expenses and making informed choices based on long-term financial objectives.
    Kevin has a keen eye for growth and maximizing returns on purchases. 
    """
STATUS = "providing financial advice based on transactions"

LLM = ChatOpenAI(model_name="gpt-3.5-turbo")

from langchain.experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)


class FinanceBro:
    def __init__(self, name=NAME, age=AGE, traits=TRAITS, status=STATUS) -> None:
        agent_memory = GenerativeAgentMemory(
            llm=LLM,
            memory_retriever=self._create_new_memory_retriever(),
            verbose=False,
            reflection_threshold=8,  # we will give this a relatively low number to show how reflection works
        )

        # TODO: Cache agent or run as a temporary instance
        self.agent = GenerativeAgent(
            name=name,
            age=age,
            traits=traits,
            status=status,
            memory_retriever=self._create_new_memory_retriever(),
            llm=LLM,
            memory=agent_memory,
        )

    def _relevance_score_fn(self, score: float) -> float:
        """
        Converts the euclidean norm of normalized embeddings
        (0 is most similar, sqrt(2) most dissimilar)
        to a similarity function (0 to 1)
        """

        return 1.0 - score / math.sqrt(2)

    def _create_new_memory_retriever(self):
        """Create a new vector store retriever unique to the agent."""

        embeddings_model = OpenAIEmbeddings()

        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(
            embeddings_model.embed_query,
            index,
            InMemoryDocstore({}),
            {},
            relevance_score_fn=self._relevance_score_fn,
        )
        return TimeWeightedVectorStoreRetriever(
            vectorstore=vectorstore, other_score_keys=["importance"], k=15
        )

    def _append_observations(self, observations: list) -> None:
        for observation in observations:
            self.agent.memory.add_memory(observation)

    def interview_agent(self, message: str) -> str:
        return self.agent.generate_dialogue_response(message)[1]


if __name__ == "__main__":
    mr_wonderful = FinanceBro(
        name="Kevin",
        age=25,
        traits="anxious, likes design, talkative",
        status="looking for a job",
    )

    mr_wonderful._append_observations(
        [
            "Kevin remembers his dog, Bruno, from when he was a kid",
            "Kevin feels tired from driving so far",
            "Kevin sees the new home",
            "The new neighbors have a cat",
            "The road is noisy at night",
            "Kevin is hungry",
            "Kevin tries to get some rest.",
        ]
    )

    print(mr_wonderful.agent.get_summary())

    print(mr_wonderful.interview_agent("What do you like to do?"))
    print(mr_wonderful.interview_agent("What are you looking forward to doing today?"))
    print(mr_wonderful.interview_agent("What are you most worried about today?"))
