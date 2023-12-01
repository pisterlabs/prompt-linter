import streamlit as st
from streamlit_chat import message

from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.chains import VectorDBQAWithSourcesChain
from lib import schema, vdb
from sqlalchemy import create_engine


DATABASES = ['adventure_works']


def app():
    st.set_page_config(page_title="Postgres Assistant", page_icon=":robot:")
    st.header("Postgres Assistant")

    if 'db_name' not in st.session_state:
        st.session_state['db_name'] = None

    if 'db_chain' not in st.session_state:
        st.session_state['db_chain'] = None

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    st.title('Database Assistant')
    with st.sidebar.expander(label='Database Setting'):
        db_name = st.selectbox('Select your database', DATABASES)
        if st.session_state['db_chain'] is None or st.session_state['db_name'] != db_name:
            db = SQLDatabase.from_uri(
                f'postgresql://postgres:postgres@localhost/{db_name}')
            llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo')
            st.session_state['db_name'] = db_name
            st.session_state['db_chain'] = SQLDatabaseChain(
                llm=llm, database=db, verbose=True)

    if not st.session_state['db_chain']:
        st.write(
            f'Cannot load {db_name}. Please check your database connection or try another database')
        return

    st.write(f'Database {db_name} loaded. Ready to answer questions')

    input = st.text_area('Please ask questions for the database',  key='input')
    if st.button('Ask'):
        with st.spinner('Generating answer...'):
            chain = st.session_state['db_chain']
            chain.run(input)
            output = db_chain = SQLDatabaseChain(
                llm=llm, database=db, verbose=True)

            st.session_state["generated"].append(output)
            st.session_state['past'].append(input)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i],
                    is_user=True, key=str(i) + "_user")


if __name__ == "__main__":
    app()
