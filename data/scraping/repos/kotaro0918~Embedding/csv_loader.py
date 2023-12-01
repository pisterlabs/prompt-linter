# Embedding用
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
# Vector 格納 / FAISS
from langchain.vectorstores import FAISS
# テキストファイルを読み込む
from langchain.document_loaders import TextLoader
# Q&A用Chain
from langchain.chains.question_answering import load_qa_chain
# ChatOpenAI GPT 3.5
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    # メッセージテンプレート
    ChatPromptTemplate,
    # System メッセージテンプレート
    SystemMessagePromptTemplate,
    # assistant メッセージテンプレート
    AIMessagePromptTemplate,
    # user メッセージテンプレート
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    # それぞれ GPT-3.5-turbo API の assistant, user, system role に対応
    AIMessage,
    HumanMessage,
    SystemMessage
)
target_input=input()
from langchain.document_loaders import CSVLoader
 
# CSVLoaderを使用してCSVファイルからデータを読み込む
loader = CSVLoader("doc_class.csv")
documents = loader.load()

elements=[]
# 各ドキュメントのコンテンツとメタデータにアクセスする
for document in documents:
    content = document.page_content
    elements.append(content)

embeddings = OpenAIEmbeddings()

db = FAISS.from_texts(elements, embeddings)
query = f"""あなたは図書館で書籍の分類を担当する司書です。

以下の text はある本の概要説明文です。地名に注目してこの説明文の内容に適切な分類項目をidに含まれる数字で提示してください。
答えに至る過程も出力してください
text: {target_input}"""
embedding_vector = embeddings.embed_query(query)
docs_and_scores = db.similarity_search_by_vector(embedding_vector)

print(len(embedding_vector))
from langchain.callbacks import get_openai_callback
# load_qa_chainを準備

with get_openai_callback() as cb:
    chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff", verbose=True)

# 質問応答の実行
    print(chain({"input_documents": docs_and_scores, "question": query},return_only_outputs=True))
    print(cb)
