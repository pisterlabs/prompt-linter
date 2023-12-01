from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
from qdrant_db import QdrantSingleton

# import database as db



# from langchain.llms import OpenAI
# from langchain.agents import AgentType, initialize_agent, load_tools



# ########################################################################
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     AIMessagePromptTemplate,
#     HumanMessagePromptTemplate,
# )
# from langchain.schema import AIMessage, HumanMessage, SystemMessage
# from langchain.callbacks import get_openai_callback
# OPENAI_API_KEY = "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"
# # Initialize the ChatOpenAI object
# chat = ChatOpenAI(temperature=0.1, openai_api_key=OPENAI_API_KEY)

# # Define the system message to set the role of the AI
# system_message_content = "You are a higher education professor specialized in explaining complex topics to students step-by-step. Dont worry about being too detailed, dont focus on the answer, focus on the process on how you got to the answer."
# system_message_prompt = SystemMessagePromptTemplate.from_template(system_message_content)

# # Define a template for human messages
# human_template = "Subject: {subject}, Topic: {topic}, Question: {specific_question}"
# human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# # Combine them into a ChatPromptTemplate
# chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# # Example student query
# student_query = {
#     "subject": "Mathematics",
#     "topic": "Calculus",
#     "specific_question": "How do I integrate x^2?"
# }



########################################################################
def thoth_page():
    # st_callback = StreamlitCallbackHandler(st.container())
    # from book_notes.book_note_page import initialize_qdrant_client, get_vector_store
    # Initialization
        
    qdrant_singleton_retrival = QdrantSingleton()
    vector_store = qdrant_singleton_retrival.get_vector_store(st.session_state['username'])


    #plug vector search
    # from langchain.chains import RetrievalQA
    # Get an OpenAI API Key before continuing
    
        
        
    # qa = RetrievalQA.from_chain_type(
    #     llm=ChatOpenAI(openai_api_key=openai_api_key),
    #     chain_type="stuff",
    #     retriever= vector_store.as_retriever(),
    # )

    query = st.text_input("Enter a query")
    if query:
        response3 = vector_store.similarity_search(query,k=1)
        st.write(response3)

########################################################################
    # with st.sidebar:
    #     st.radio("Select a language model", ["Lawyer Thoth", "Math Thoth"])
    #     st.subheader("knowledge base")
    #     db_files = db.list_files(st.session_state['username'])
    #     for file in db_files:
    #         st.write(file)
            
    # llm = OpenAI(temperature=0, streaming=True)
    # tools = load_tools(["ddg-search"])
    # agent = initialize_agent(
    #     tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    # )

    # if prompt := st.chat_input():
    #     st.chat_message("user",).write(prompt)
    #     with st.chat_message("assistant"):
    #         st_callback = StreamlitCallbackHandler(st.container())
    #         response = agent.run(prompt, callbacks=[st_callback])
    #         st.write(response)