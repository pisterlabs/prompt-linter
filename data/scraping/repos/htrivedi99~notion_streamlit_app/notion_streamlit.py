import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List
import streamlit as st
import tiktoken
from streamlit_chat import message


BASE_URL = "https://api.notion.com"


def notion_get_blocks(page_id: str, headers: dict):
    res = requests.get(f"{BASE_URL}/v1/blocks/{page_id}/children?page_size=100", headers=headers)
    return res.json()


def notion_search(query: dict, headers: dict):
    res = requests.post(f"{BASE_URL}/v1/search", headers=headers, data=query)
    return res.json()


def get_page_text(page_id: str, headers: dict):
    page_text = []
    blocks = notion_get_blocks(page_id, headers)
    for item in blocks['results']:
        item_type = item.get('type')
        content = item.get(item_type)
        if content.get('rich_text'):
            for text in content.get('rich_text'):
                plain_text = text.get('plain_text')
                page_text.append(plain_text)
    return page_text


def load_notion(headers: dict) -> list:
    documents = []
    all_notion_documents = notion_search({}, headers)
    items = all_notion_documents.get('results')
    for item in items:
        object_type = item.get('object')
        object_id = item.get('id')
        url = item.get('url')
        title = ""
        page_text = []

        if object_type == 'page':
            title_content = item.get('properties').get('title')
            if title_content:
                title = title_content.get('title')[0].get('text').get('content')
            elif item.get('properties').get('Name'):
                if len(item.get('properties').get('Name').get('title')) > 0:
                    title = item.get('properties').get('Name').get('title')[0].get('text').get('content')

            page_text.append([title])
            page_content = get_page_text(object_id, headers)
            page_text.append(page_content)

            flat_list = [item for sublist in page_text for item in sublist]
            text_per_page = ". ".join(flat_list)
            if len(text_per_page) > 0:
                documents.append(text_per_page)

    return documents


def chunk_tokens(text: str, token_limit: int) -> list:
    tokenizer = tiktoken.get_encoding(
        "cl100k_base"
    )

    chunks = []
    tokens = tokenizer.encode(text, disallowed_special=())

    while tokens:
        chunk = tokens[:token_limit]
        chunk_text = tokenizer.decode(chunk)
        last_punctuation = max(
            chunk_text.rfind("."),
            chunk_text.rfind("?"),
            chunk_text.rfind("!"),
            chunk_text.rfind("\n"),
        )
        if last_punctuation != -1:
            chunk_text = chunk_text[: last_punctuation + 1]
        cleaned_text = chunk_text.replace("\n", " ").strip()

        if cleaned_text and (not cleaned_text.isspace()):
            chunks.append(cleaned_text)
        tokens = tokens[len(tokenizer.encode(chunk_text, disallowed_special=())):]

    return chunks


def load_data_into_vectorstore(client, docs: List[str]):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    qdrant_client = Qdrant(client=client, collection_name="notion_streamlit", embedding_function=embeddings.embed_query)
    ids = qdrant_client.add_texts(docs)
    return ids


@st.cache_resource
def connect_to_vectorstore():
    client = QdrantClient(host="localhost", port=6333, path="/path/to/qdrant/qdrant_storage")
    try:
        client.get_collection("notion_streamlit")
    except Exception as e:
        client.recreate_collection(
            collection_name="notion_streamlit",
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        )
    return client


@st.cache_data
def cache_headers(notion_api_key: str):
    headers = {"Authorization": f"Bearer {notion_api_key}", "Content-Type": "application/json",
               "Notion-Version": "2022-06-28"}
    return headers


@st.cache_resource
def load_chain(_client, api_key: str):
    if len(api_key) == 0:
        api_key = "temp value"
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = Qdrant(client=_client, collection_name="notion_streamlit", embedding_function=embeddings.embed_query)
    chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo',
            openai_api_key=api_key),
            retriever=vectorstore.as_retriever()
    )
    return chain


st.title('Chat With Your Notion Documents!')

vector_store = connect_to_vectorstore()
with st.sidebar:
    openai_api_key = st.text_input(label='#### Your OpenAI API Key', placeholder="Paste your OpenAI API key here", type="password")
    notion_api_key = st.text_input(label='#### Your Notion API Key', placeholder="Paste your Notion API key here",
                                 type="password")

    notion_headers = cache_headers(notion_api_key)

    load_data = st.button('Load Data')
    if load_data:
        documents = load_notion(notion_headers)

        chunks = []
        for doc in documents:
            chunks.extend(chunk_tokens(doc, 100))

        for chunk in chunks:
            print(chunk)

        load_data_into_vectorstore(vector_store, chunks)
        print("Documents loaded.")

chain = load_chain(vector_store, openai_api_key)

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


user_input = st.text_input("You: ", placeholder="Chat with your notion docs here 👇", key="input")

if user_input:
    result = chain({"question": user_input, "chat_history": st.session_state["generated"]})
    response = result['answer']

    st.session_state['past'].append(user_input)
    st.session_state['generated'].append((user_input, result["answer"]))

if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i][1], key=str(i))



