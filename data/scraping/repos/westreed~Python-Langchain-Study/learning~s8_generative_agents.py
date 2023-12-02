from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS

from langchain.experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)

from key import APIKEY
import math
import faiss

"""
패키지 설치
pip3 install torch torchvision torchaudio
pip install faiss-cpu -q
pip install tiktoken

둘다, gpu 없이 cpu only로 설치.
"""


def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    # 이것은 몇 가지 요소에 따라 달라집니다:
    # - VectorStore에서 사용하는 거리/유사성 측정 방법
    # - 임베딩의 스케일(OpenAI의 경우 단위 노름이지만 다른 경우는 아님)
    # 이 함수는 정규화된 임베딩의 유클리드 노름(0이 가장 유사하고 sqrt(2)가 가장 다른 경우)을
    # relevance_score_fn(0에서 1까지)로 변환합니다.
    return 1.0 - score / math.sqrt(2)


def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn,
    )
    # vectorstore
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=15
    )


if __name__ == "__main__":
    KEY = APIKEY()
    LLM = ChatOpenAI(openai_api_key=KEY.openai_api_key)

    jsh_memory = GenerativeAgentMemory(
        llm=LLM,
        memory_retriever=create_new_memory_retriever(),
        verbose=False,
        reflection_threshold=8,  # we will give this a relatively low number to show how reflection works
    )

    jsh = GenerativeAgent(
        name="세훈",
        age=28,
        traits="걱정이 많은, 개발을 좋아하는, 과묵한",  # You can add more persistent traits here
        status="어떤 소프트웨어를 개발할지 고민 중",  # When connected to a virtual world, we can have the characters update their status
        memory_retriever=create_new_memory_retriever(),
        llm=LLM,
        memory=jsh_memory,
    )

    jsh_observations = [
        "세훈은 아침에 일찍 일어나는게 힘듭니다.",
        "세훈은 IT와 소프트웨어 개발을 좋아합니다.",
        "세훈은 IT기기에 대한 관심이 많습니다.",
        "세훈은 게임을 좋아합니다.",
        "세훈은 혼자서 서울에 자취하고 있습니다.",
        "세훈은 지금 배가 고픕니다.",
        "세훈은 SW 마에스트로에 합격하여 연수과정을 받고 있습니다.",
    ]
    for observation in jsh_observations:
        jsh.memory.add_memory(observation)

    print(jsh_memory.memory_retriever)
    print(jsh.get_summary())


    def interview_agent(agent: GenerativeAgent, message: str) -> str:
        """Help the notebook user interact with the agent."""
        new_message = f"상대방이 `{message}`라고 말합니다."
        return agent.generate_dialogue_response(new_message)[1]

    res = interview_agent(jsh, "넌 무엇을 좋아하니?")
    print(res)