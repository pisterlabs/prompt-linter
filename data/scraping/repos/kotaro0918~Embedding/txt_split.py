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
title_input=input()
target_input=input()
loader = TextLoader('doc_class.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(docs, embeddings)
query = f"""以下の文章は地域の暮らしに関して書かれた本の解説とそのタイトルです。この情報をもとにこの本に適した分類項目の候補を3つ数字で示してください
根拠と一緒に出力してください
title: {title_input}
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
