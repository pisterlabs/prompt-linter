# import streamlit module
import streamlit as st

# import langchain modules
from langchain.document_loaders import ConfluenceLoader
from langchain.indexes import VectorstoreIndexCreator

# define method generate_response
def generate_response(query_text):
    loader = ConfluenceLoader(
                url="https://inauman.atlassian.net/wiki",
                username=st.secrets.confluence.username,
                api_key=st.secrets.confluence.apikey
            )
    documents = loader.load(space_key=st.secrets.confluence.space,limit=50)

    # create index from documents
    index = VectorstoreIndexCreator().from_documents(documents)

    # Question-answering
    response = index.query(query_text)
    return response

# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')

# Title
st.title('🦜🔗 Ask the Doc App')

# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.')

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    
    submitted = st.form_submit_button('Submit', disabled=not(query_text))
    if submitted:
        with st.spinner('Calculating...'):
            response = generate_response(query_text)
            result.append(response)

if len(result):
    st.info(response)