import pandas as pd
import streamlit as st

import cohere, os
from qdrant_client import QdrantClient

from langchain.schema import Document
from langchain.vectorstores import Qdrant
from langchain.embeddings import CohereEmbeddings

from dotenv import load_dotenv
load_dotenv()

def collection_management(FACTS_VDB):
    # Define basic params
    client, embeddings = define_globals(FACTS_VDB)
    # Read and call saved collections by username
    st.session_state.collection_namelist = [collection.name 
                                            for collection in client.get_collections().collections 
                                            if collection.name.startswith(st.session_state.user_email.split('@')[0])
                                            ]

    collection_name = st.selectbox('Select Collection:', 
                                   st.session_state.collection_namelist, 
                                   key='rv_collection_dropdown')
    
    # Read single collection
    read_single_collection(client, collection_name)

# Define GLOBALS ---------------------------------------------------------------
def define_globals(FACTS_VDB=None):
    embeddings = CohereEmbeddings(model="multilingual-22-12", 
                                cohere_api_key=os.environ['COHERE_API_KEY'])
    qdrant_url = os.environ['QDRANT_URL']
    qdrant_api_key = os.environ['QDRANT_API_KEY']
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    return client, embeddings
    
# Call Collection ---------------------------------------------------------------
def read_single_collection(client, collection_name):
    try:
        n_points = client.count(collection_name=collection_name).count
    except:
        st.info('No available documents on-cloud')
        st.stop()
        pass
    if n_points > 0:
        # Docs selection
        doc_id = st.number_input(label='Document ID:', 
                                        min_value=0, max_value=n_points-1,
                                        value=0, key='doc_id')
        # Display document by ID
        point = client.retrieve(collection_name=collection_name, ids=[doc_id])
        st.warning(point[0].payload['page_content'].replace('_', ' '))
        # Display metadata of document
        metadata = point[0].payload['metadata']
        try:
            if len(metadata) > 0:
                st.info(f'Docs information:\n {metadata}')
        except TypeError:
            pass
    else:
        st.info('No available documents')
    pass
            
def remove_collection(FACTS_VDB):
    # Define basic params
    client, embeddings = define_globals(FACTS_VDB)
    # Define removing collection name
    collection_name = st.selectbox('Select Collection:', 
                                   st.session_state.collection_namelist, 
                                   key='remove_collection_dropdown')
    
    # Button to delete a collection
    if st.button('Delete Collection', key='rm_collection_btn'):
        client.delete_collection(collection_name=st.session_state.remove_collection_dropdown)
        st.experimental_rerun()

def title_ui(title):
    st.write(f'''<div style="text-align: center;">
                <h5 style="color: #1D5B79;
                            font-size: 40px;
                            font-weight: bold;
                ">{title}</h5>
                </div>
            ''',
                unsafe_allow_html=True)
    

    
