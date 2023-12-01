import langchain
from dotenv import load_dotenv
from langchain.chains import HypotheticalDocumentEmbedder, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

langchain.debug = True

load_dotenv()

# HyDE (LLMが生成した仮説的な回答のベクトル化) の準備
base_embeddings = OpenAIEmbeddings()
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
embeddings = HypotheticalDocumentEmbedder.from_llm(chat, base_embeddings, "web_search")

# FAISSで保存されたベクトルを読み込む
db = FAISS.load_local("./tmp/faiss", embeddings)
retriever = db.as_retriever()

# 「関連する文書を検索 => LLMに回答を生成させる」を実行する「RetrievalQA」を準備
qa_chain = RetrievalQA.from_chain_type(
    llm=chat, chain_type="stuff", retriever=retriever
)

query = "LangChainとは"
result = qa_chain.run(query)
print(result)
