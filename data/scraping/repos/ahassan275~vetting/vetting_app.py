import streamlit as st
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import OpenAI
from langchain.utilities import DuckDuckGoSearchAPIWrapper
import re
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import pathlib
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
import openai
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field
from vetting_questions import extracted_dict_list
from langchain.agents import initialize_agent
from langchain.agents import AgentType
import random
import os

# from streamlit_agent.callbacks.capturing_callback_handler import playback_callbacks
# from streamlit_agent.clear_results import with_clear_container

#
# openai_api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = os.environ["OPENAI_API_KEY"]

MAX_API_CALLS = 25  # set your limit

# Initialize count of API calls
if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0


def chat_with_agent(input_text):
    response = agent({"input": input_text})
    return response['output']


class DocumentInput(BaseModel):
    question: str = Field()


llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo-16k")

tools = []
files = [
    {
        "name": "dedoose-terms-of-service",
        "path": "TERMS OF SERVICE.pdf",
    },
]

for file in files:
    loader = PyPDFLoader(file["path"])
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings()
    retriever = FAISS.from_documents(docs, embeddings).as_retriever()

    # Wrap retrievers in a Tool
    tools.append(
        Tool(
            args_schema=DocumentInput,
            name=file["name"],
            description=f"useful when you want to answer questions about {file['name']}",
            func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        )
    )

llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
)

agent = initialize_agent(
    agent=AgentType.OPENAI_FUNCTIONS,
    tools=tools,
    llm=llm,
    verbose=True,
)

st.set_page_config(page_title="Vetting Assistant")

st.title("Vetting Assistant")

for question_dict in extracted_dict_list:
    user_input = question_dict['question']
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_input, callbacks=[st_callback])
        st.write(response)

# Initialize chat_history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "run_count" not in st.session_state:
    st.session_state.run_count = 0

if st.button('Start'):
    if st.session_state.run_count >= 1:
        st.warning("You have reached the maximum number of runs for this session.")
    else:
        st.session_state.run_count += 1

        # Select 3 random questions
        selected_questions = random.sample(extracted_dict_list, 3)

        # Loop over selected questions
        for question_dict in selected_questions:
            user_input = question_dict['question']

            # Save user's message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                response = agent.run(user_input, callbacks=[st_callback])
                st.write(response)

                # Save assistant's response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})


# for question in extracted_dict_list:
#     input_text = question['question']
#     response = chat_with_agent(input_text)
#     print(f"Question: {input_text}")
#     print(f"Response: {response}")
#     print()
