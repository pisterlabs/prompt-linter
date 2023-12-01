__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
import logging
from typing import List
from chromadb.utils.embedding_functions import *
from chromadb.api.models.Collection import Collection
from chromadb import Settings

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import uuid
import pandas as pd


logging.basicConfig(level=logging.DEBUG)
view_collection = None

def get_chroma_client(host: str, port: int) -> chromadb.HttpClient:
    try:
        # chroma_client = chromadb.HttpClient(host=host,
        #                                     port=port,
        #                                     ssl=False,
        #                                     settings=Settings(anonymized_telemetry=False))
        chroma_client = chromadb.PersistentClient(st.session_state["CHROMA_PATH"])
        return chroma_client
    except Exception as ex:
        st.toast(body="Failed to establish connection",
                 icon="😢")
        logging.error(f"Error: {str(ex)}")

def get_chroma_collection_names(chroma_client: chromadb.HttpClient) -> List:
    collection_names = []
    collections = chroma_client.list_collections()
    for col in collections:
        collection_names.append(col.name)

    return collection_names

def create_collection(chroma_client: chromadb.HttpClient,
                      collection_name: str,
                      embedding_function: str):

    collection_map = {
        "VertexEmbedding": GoogleVertexEmbeddingFunction
    }

    try:
        chroma_client.create_collection(name=collection_name,
                                        embedding_function=collection_map[embedding_function])
    except Exception as ex:
        st.toast(body="Failed to create new connection",
                icon="😢")
        logging.error(f"Error: {str(ex)}")

def delete_selected_collection(chroma_client: chromadb.HttpClient,
                      collection_name: str):

    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception as ex:
        st.toast(body="Failed to delete connection",
                icon="😢")
        logging.error(f"Error: {str(ex)}")

def upload_chromadb(collection: Collection, file_paths: str):
    try:
        loader = DirectoryLoader(st.session_state["UPLOAD_FOLDER"], glob="*.*")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        for doc in docs:
            fid = f"{str(uuid.uuid4())}"
            embedding = collection._embedding_function([doc.page_content])
            source = doc.metadata['source'].split("/")[-1]
            collection.add(ids=[fid],
                           metadatas={'source': source},
                           documents=doc.page_content,
                           embeddings=embedding)

        for file_path in file_paths:
            os.remove(file_path)

        st.toast(body='New file uploaded!',
                icon='✅')
    except Exception as ex:
        raise ex


def get_collection_data(collection: Collection):
    return collection.get(
        include=["documents", "embeddings", "metadatas"]
    )

def get_vis(df: pd.DataFrame):
    import plotly.express as px
    from sklearn.decomposition import PCA

    embeddings = df['embeddings'].tolist()
    source = df['metadatas'].tolist()
    cols = [str(i) for i in range(len(embeddings[0]))]
    new_data = {}
    new_data['source'] = [s['source'] for s in source]
    for col in cols:
        new_data[col] = []

    for embedding in embeddings:
        for idx, value in enumerate(embedding):
            new_data[str(idx)].append(value)

    embedding_df = pd.DataFrame(data=new_data)

    X = embedding_df[cols]

    pca = PCA(n_components=3)
    components = pca.fit_transform(X)

    return px.scatter_3d(
        components, x=0, y=1, z=2,
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'},
        color=embedding_df['source']
    )


vector_database_host = st.sidebar.text_input(label='Chroma host', value="localhost")
vector_database_port = st.sidebar.text_input(label='Chroma port', value=8000)

if st.sidebar.button("Connect"):
    st.session_state["vector_database_host"] = vector_database_host
    st.session_state["vector_database_port"] = vector_database_port
    st.session_state["vector_database_client"] = get_chroma_client(host=st.session_state["vector_database_host"],
                                                                   port=st.session_state["vector_database_port"])
    logging.info("Chroma client connection established!")

    st.session_state["is_connected"] = True

    st.toast(body='Connection established!',
             icon='✅')

if "is_connected" in st.session_state and st.session_state["is_connected"]:

    new_collection_name = st.sidebar.text_input(label='New collection name', placeholder='')
    new_collection_embedding_function = st.sidebar.selectbox(label='New collection embeeding function',
                                                             options=["VertexEmbedding"])

    if st.sidebar.button("Create"):
        st.session_state["new_collection_name"] = new_collection_name
        st.session_state["new_collection_embedding_function"] = new_collection_embedding_function

        create_collection(chroma_client=st.session_state["vector_database_client"],
                                                                       collection_name=st.session_state["new_collection_name"],
                                                                       embedding_function=st.session_state["new_collection_embedding_function"])

        logging.info(f"New collection `{st.session_state['new_collection_name']}` created!")

        st.toast(body='New collection created!',
                icon='✅')


    st.session_state['collection_options'] = get_chroma_collection_names(st.session_state["vector_database_client"])

    view_placeholder = st.sidebar.empty()

    view_collection = view_placeholder.selectbox(
        label="Select a collection to view",
        options=st.session_state['collection_options'],
        key="view_selectbox"
    )

    if st.sidebar.button(label="Delete collection", type="primary"):
        delete_selected_collection(st.session_state["vector_database_client"], view_collection)
        view_placeholder.empty()

        st.session_state['collection_options'] = list(filter(lambda x: (x != view_collection), st.session_state['collection_options']))

        view_collection = view_placeholder.selectbox(
            label="Select collection to view",
            options=st.session_state['collection_options'],
            key="new_view_selectbox"
        )


if "vector_database_client" not in st.session_state:
    st.markdown('## Please connect database')
elif not view_collection:
    st.markdown('## Please create/select a collection')
else:
    collection = st.session_state["vector_database_client"].get_collection(name=view_collection)
    df = pd.DataFrame(get_collection_data(collection))

    st.markdown("### Dataframe")
    dataframe_placeholder = st.empty()
    dataframe_placeholder.data_editor(df)

    st.markdown("### Document Upload")

    file_uploads = st.file_uploader('Only PDF(s)', type=['pdf'], accept_multiple_files=True)
    file_paths = []


    st.markdown("### PCA Visualisation")
    pca_placeholder = st.empty()
    if len(df):
        pca_placeholder.plotly_chart(get_vis(df), theme="streamlit", use_container_width=True)


    if file_uploads:
        for pdf_file in file_uploads:
            file_path = os.path.join(st.session_state["UPLOAD_FOLDER"], pdf_file.name)
            file_paths.append(file_path)

            with open(file_path,"wb") as f:
                f.write(pdf_file.getbuffer())

        upload_chromadb(collection, file_paths)
        dataframe_placeholder.empty()
        new_df = pd.DataFrame(get_collection_data(collection))
        dataframe_placeholder.data_editor(new_df)
        pca_placeholder.empty()
        pca_placeholder.plotly_chart(get_vis(new_df), theme="streamlit", use_container_width=True)
