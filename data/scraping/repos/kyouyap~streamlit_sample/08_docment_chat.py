import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import datetime
import dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
import os
import openai
import glob
from typing import Any

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)


class JapaneseCharacterTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, **kwargs: Any):
        separators = ["\n\n", "\n", "。", "、", " ", ""]
        super().__init__(separators=separators, **kwargs)


def init():
    dotenv.load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # TODO:　必要に応じて追記する


def load_qdrant():
    """
    Qdrantをロードする関数。
    """
    QDRANT_PATH, COLLECTION_NAME = os.getenv("QDRANT_PATH"), os.getenv(
        "COLLECTION_NAME"
    )
    client = QdrantClient(path=QDRANT_PATH)

    # すべてのコレクション名を取得
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    # コレクションが存在しなければ作成
    if COLLECTION_NAME not in collection_names:
        # コレクションが存在しない場合、新しく作成します
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print("collection created")

    return Qdrant(
        client=client, collection_name=COLLECTION_NAME, embeddings=OpenAIEmbeddings()
    )


def init_page():
    st.set_page_config(page_title="My Great ChatGPT", page_icon="🤗")
    st.header("My Great ChatGPT 🤗")
    st.sidebar.title("Options")


def init_messages():
    init_content = f"""
    You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture. Knowledge cutoff: 2021-09. Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}.
    """
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=init_content)]
        st.session_state.costs = []


def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"

    # サイドバーにスライダーを追加し、temperatureを0から2までの範囲で選択可能にする
    # 初期値は0.0、刻み幅は0.1とする
    temperature = st.sidebar.slider(
        "Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.01
    )
    st.session_state.model_name = model_name
    return ChatOpenAI(temperature=temperature, model_name=model_name, streaming=True)


def show_massages(messages_container):
    messages = st.session_state.get("messages", [])
    with messages_container:
        for message in messages:
            if isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            else:  # isinstance(message, SystemMessage):
                st.write(f"System message: {message.content}")


def build_qa_model(llm):  # noqa
    """
    質問応答モデルを構築する関数。
    """
    qdrant = load_qdrant()
    retriever = qdrant.as_retriever()
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer in Japanese:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
    )
    return qa


def query_search(query):  # noqa
    """
    質問応答モデルを構築する関数。
    """
    qdrant = load_qdrant()
    print("query_search")
    # qdrant.similarity_search_with_score("「フリーランスのリモートワークの実態」について教えて。", k=2)
    docs = qdrant.similarity_search_with_score(query, k=2)
    return docs


def place_input_form(input_container, messages_container, llm):
    messages = st.session_state.get("messages", [])
    with input_container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_area(label="Message: ", key="input")
            submit_button = st.form_submit_button(label="Send")
        if submit_button and user_input:
            # 何か入力されて Submit ボタンが押されたら実行される
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("ChatGPT is typing ..."):
                response = build_qa_model(llm).run(user_input)
            st.session_state.messages.append(AIMessage(content=response))
            st.session_state.messages = st.session_state.messages[-3:]


def build_vector_store():
    qdrant = load_qdrant()
    text_files = glob.glob("documents/*.txt", recursive=True)
    print(text_files)
    docs = []
    for text_file in text_files:
        with open(text_file) as f:
            text = f.read()
        text_splitter = CharacterTextSplitter(
            separator="\n\n",  # 文章を分割する文字列
            chunk_size=1800,  # チャンクの文字数
            chunk_overlap=0,  # チャンク間で重複させる文字数
        )
        split_texts = text_splitter.split_text(text)
        docs.extend([Document(page_content=split_text) for split_text in split_texts])
    qdrant.add_documents(docs)
    print(docs)


def document_to_vector():
    st.write("docment配下のファイルをベクトル化します")
    submit_button = st.button(label="To vector")
    if submit_button:
        load_qdrant()
        build_vector_store()


def chat_with_gpt():
    llm = select_model()
    messages_container = st.container()
    input_container = st.container()

    place_input_form(input_container, messages_container, llm)
    show_massages(messages_container)


def main():
    init()
    init_page()
    init_messages()
    selection = st.sidebar.radio("Go to", ["Document to vector", "Chat"])
    if selection == "Document to vector":
        document_to_vector()
    else:
        chat_with_gpt()


if __name__ == "__main__":
    main()
