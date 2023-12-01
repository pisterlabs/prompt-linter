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
from langchain.document_loaders import UnstructuredFileLoader
target_input=input()
loader = UnstructuredFileLoader("doc_class.pdf",mode="elements")
docs = loader.load()
print(f"number of docs: {len(docs)}")
embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(docs, embeddings)
query = f"""以下の文章は本の解説です。この情報をもとにこの本に適した分類項目を三桁の数字で示してください
text: {target_input}"""
embedding_vector = embeddings.embed_query(query)
docs_and_scores = db.similarity_search_by_vector(embedding_vector)

print(len(embedding_vector))
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
# load_qa_chainを準備
    chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff", verbose=True)

# 質問応答の実行
    print(chain({"input_documents": docs_and_scores, "question": query},return_only_outputs=True))
    print(cb)