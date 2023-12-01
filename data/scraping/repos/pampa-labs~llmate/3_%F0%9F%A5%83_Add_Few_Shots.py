import json

import llmate_config
import streamlit as st
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_types import AgentType
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS

from utils import update_agent

llmate_config.general_config()


def include_few_shots():
    few_shots = st.session_state["few_shots"]

    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state['openai_api_key'])

    few_shot_docs = [
        Document(
            page_content=example["question"],
            metadata={"sql_query": example["sql_query"]},
            )
        for example in few_shots
    ]
    vector_db = FAISS.from_documents(few_shot_docs, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    st.session_state['few_shot_retriever'] = retriever

    description = """
    This tool will help you understand similar examples to adapt them to the user question.
    Input to this tool should be the user question.
    """

    retriever_tool = create_retriever_tool(
        retriever, name="sql_get_similar_examples", description=description
    )

    st.session_state[
        "sql_agent_suffix"
    ] = "Always use the 'sql_get_similar_examples' tool before using any other tool."

    st.session_state['extra_tools'] = [retriever_tool]

    update_agent()


if ('openai_api_key' not in st.session_state) or (st.session_state['openai_api_key'] == ''):
    st.error('Please load OpenAI API KEY and connect to a database', icon='🚨')
else:
    st.subheader("Add few shot examples")
    st.markdown(
        """
    If your agent is having trouble answering some complex questions, giving it some concrete examples might work.
    
    In fact, adding few shot examples to your prompt has been [proven](https://arxiv.org/abs/2204.00498) to improve accuracy significantly when dealing with hard questions.
    
    Few shot examples 👇
 """
    )

    # uploaded_few_shots = st.file_uploader(
    #     "Please upload a few shot dataset (.json): ",
    #     type=["json"],
    #     accept_multiple_files=False,
    # )
    if "few_shots" not in st.session_state:
        with open(st.session_state['database_options'][st.session_state['selected_database']]['few_shots'], "r") as file:
            st.session_state["few_shots"] = json.load(file)

    # if uploaded_few_shots:
        # st.session_state["few_shots"] = json.loads(uploaded_few_shots.read())
        # st.session_state["few_shots.name"] = st.session_state['few_shots'].name 
        st.session_state["few_shots.name"] = "Example_chinook"


    if 'few_shots' in st.session_state:
        edited_data = st.data_editor(
            st.session_state["few_shots"],
            num_rows="dynamic",
            key="few_shots_editor",
            use_container_width=True,
            height=300,
            on_change=include_few_shots,
        )
        if 'few_shot_retriever' not in st.session_state:
            include_few_shots()
        test_few_shot = st.text_input("Test which few shots are retriever from a specific question:")
        
        if test_few_shot:
            response = st.session_state['few_shot_retriever'].get_relevant_documents(test_few_shot)
            resp = [{'question':doc.page_content,'query':doc.metadata['sql_query']} for doc in response]
            st.dataframe(resp)