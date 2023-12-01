import langchain
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS

langchain.verbose = True

load_dotenv()

# ChatGPTで生成した架空の人名とプロフィール
texts = [
    """Name: Zephyrina Bluemoon
Profile: Zephyrina Bluemoon is an astrophysicist who was awarded the Nobel Prize in Physics in 2040. His research on dark matter and multidimensional universes has led to the development of a new cosmological theory.
""",
    """Name: Quill Solstice
Profile: Quill Solstice is an internationally renowned environmental activist, working on climate change and biodiversity conservation. His initiatives have received widespread support, especially among the youth around the world.
""",
    """Name: Seraphim Vortex
Profile: Seraphim Vortex is a globally acclaimed pianist, whose performances are often described as "the voice of nature". Through her classical music, she conveys a message of environmental preservation to the world.
""",
    """Name: Eclipse Stardust
Profile: Eclipse Stardust is an AI developer known for her research in autonomous drones. Her drone technology has been used in disaster rescue and environmental surveys, saving many lives.
""",
    """Name: Celestia Rainbow
Profile: Celestia Rainbow is a world-famous novelist, and her works have been translated into more than 30 languages. Her novels, characterized by a deep understanding of humanity and delicate portrayals of the human heart, have received international acclaim.
""",
]

# 用意したデータをFAISSで検索する準備
embeddings = OpenAIEmbeddings()
db = FAISS.from_texts(texts, embeddings)
faiss_retriever = db.as_retriever(search_kwargs={"k": 1})

# 用意したデータをBM25で検索する準備
bm25_retriever = BM25Retriever.from_texts(texts, k=1)

# 2つのRetrieverを組み合わせる
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)

# 「関連する文書を検索 => LLMに回答を生成させる」を実行する「RetrievalQA」を準備
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=chat, chain_type="stuff", retriever=ensemble_retriever
)

query = "Zephyrina Bluemoonさんについて教えてください。"
result = qa_chain.run(query)
print(result)
