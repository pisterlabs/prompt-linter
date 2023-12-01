import langchain
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

langchain.verbose = True

load_dotenv()

# FAISSで保存されたベクトルを読み込む
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("./tmp/faiss", embeddings)
retriever = db.as_retriever()

# LangChainにおけるRAGの基本である「RetrievalQA」を準備する
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=chat, chain_type="stuff", retriever=retriever
)

# 「クエリに関連する文書を検索 => LLMに回答を生成させる」という流れを実行する
query = "AWSのS3からデータを読み込むためのDocumentLoaderはありますか？"
result = qa_chain.run(query)
print(result)
