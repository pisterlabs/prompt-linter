import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from getpass import getpass
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()
# DeepLakeからデータを読み込み
vector_store = DeepLake(
    dataset_path=f"hub://{os.getenv('ACTIVELOOP_ACCOUNT_NAME')}/{os.getenv('DATA_SET_NAME')}",
    read_only=True,
    embedding_function=embeddings,
)
# Retrieverを定義
retriever = vector_store.as_retriever()
retriever.search_kwargs["distance_metric"] = "cos"
retriever.search_kwargs["fetch_k"] = 20
retriever.search_kwargs["maximal_marginal_relevance"] = True
retriever.search_kwargs["k"] = 20

# Chainを定義
model = ChatOpenAI(model_name="gpt-4")  # 'ada' 'gpt-3.5-turbo' 'gpt-4',
cr_chain = ConversationalRetrievalChain.from_llm(model, retriever=retriever)


def ask(question: str, chat_history: list) -> str:
    "Ask a question and return an answer."
    result = cr_chain({"question": question, "chat_history": chat_history})
    return result["answer"]

chat_history = []
while True:
    question = getpass("✏️ Please input:")

    print("You:", question)
    answer = ask(question, chat_history=chat_history)

    print("Code:", answer)
    chat_history.append((question, answer))
