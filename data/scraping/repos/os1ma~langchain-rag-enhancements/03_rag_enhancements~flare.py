import langchain
from dotenv import load_dotenv
from langchain.chains import FlareChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS

langchain.verbose = True

load_dotenv()

# FAISSで保存されたベクトルを読み込む
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("./tmp/faiss", embeddings)
retriever = db.as_retriever()

# FLAREの準備
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
flare = FlareChain.from_llm(
    llm=chat,
    retriever=retriever,
    min_prob=0.2,
)
# 回答の生成には、logprobsが使えるCompletions APIを使う
flare.response_chain.llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0,
    model_kwargs={"logprobs": 1},
)

query = "LangChainとは"
result = flare.run(query)
print(result)
