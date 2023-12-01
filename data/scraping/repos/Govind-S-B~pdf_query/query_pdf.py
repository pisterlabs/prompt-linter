import streamlit as st

from langchain.chains.question_answering import load_qa_chain

from langchain.llms import Ollama

from langchain.vectorstores import Chroma

embeddings = st.session_state.embeddings
client = st.session_state.db_client


def create_agent_chain():
    llm = Ollama(model="zephyr")
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain


def get_llm_response(query, collection):
    vectordb = Chroma(
        client=client,
        collection_name=collection,
        embedding_function=embeddings,
    )
    chain = create_agent_chain()
    matching_docs = vectordb.similarity_search(query)
    answer = chain.run(input_documents=matching_docs, question=query)
    return answer


def load_embedding_callback():
    st.session_state.load_embedding_clicked = True


st.title("Query PDF")

options = {collection.name for collection in client.list_collections()}
selected_option = st.selectbox('Select a PDF collection', options)


if 'clicked' not in st.session_state:
    st.session_state.load_embedding_clicked = False

load_embedding = st.button("Load", on_click=load_embedding_callback())
if st.session_state.load_embedding_clicked:
    collection = selected_option

    form_input = st.text_input('Enter Query')
    submit = st.button("Generate")

    if submit:
        st.write(get_llm_response(form_input, collection))
