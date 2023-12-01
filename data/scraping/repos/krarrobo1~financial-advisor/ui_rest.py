import streamlit as st


from langchain.callbacks import StreamlitCallbackHandler

from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.tools import Tool
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQA

from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
from langchain.tools import DuckDuckGoSearchRun

import os
import requests
from time import sleep

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

# AVATARS
av_us = "./user.png"  # "🦖"  #A single emoji, e.g. "🧑‍💻", "🤖", "🦖". Shortcodes are not supported.
av_ass = "./yoda.png"


def configure_retriever():
    """
    Configure and return a question-answering system with a contextual compression retriever.

    This function sets up a question-answering (QA) system that uses a contextual compression retriever
    for retrieving relevant information. It combines several components to create the QA system:

    1. Embeddings: Sentence embeddings are generated using the "multi-qa-MiniLM-L6-cos-v1" model,
       which serves as a feature representation for text.

    2. Vectorstore: The embeddings are used as a basis for similarity-based searching. A retriever
       is created with the embeddings, and the retriever is configured for similarity search with
       additional search parameters, such as 'k' for the number of similar items to retrieve.

    3. Language Model (LLM): A language model (GPT-3.5 Turbo) is used for generating text responses.
       The model is initialized with specific settings, including a temperature of 0 and a maximum token limit.

    4. Compressor: A compressor for text data is created based on the LLM, which can be used to compress
       and extract essential information from retrieved text.

    5. Contextual Compression Retriever: This component combines the compressor and vectorstore to create
       a retriever that retrieves information contextually and performs compression-based retrieval.

    6. QA: A question-answering system is created with the LLM and the compression retriever, using a specific
       chain type, such as "stuff."

    Returns:
        qa (RetrievalQA): A configured question-answering system with a contextual compression retriever.
    """

    embeddings = SentenceTransformerEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./db",
    ).as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", max_tokens=1000)
    compressor = LLMChainExtractor.from_llm(llm=llm)
    compression_retriver = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=vectorstore
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm, retriever=compression_retriver, chain_type="stuff"
    )

    return qa


def create_agent_tools():
    """
    Create a list of agent tools for information retrieval and searching.

    This function creates and returns a list of agent tools designed for information retrieval and searching.
    These tools can be used to assist an AI agent in performing various tasks, such as answering questions and
    looking up information from the web. Two primary tools are defined in the list:

    1. Document_Store Tool.
    2. Search Tool.

    Returns:
        tools (list of Tool): A list of agent tools for information retrieval and searching.

    """
    search = DuckDuckGoSearchRun()

    tools = [
        Tool(
            name="Document_Store",
            func=configure_retriever().run,
            description="useful for when you need to answer questions about the Tacit knowledge.\
                Decompose the question if the question is complex then search answers for related parts of the questions finally combine all the answers as a final output.\
                Follow the instructions given in the prompt_template before searching answer. If you could not find the answer say you didn't found",
        ),
        Tool(
            name="Search",
            func=search.run,
            description="Use this to lookup information from google search engine. Use it only after you have tried using the Document_Store tool.",
        ),
    ]

    return tools


def create_agent_prompt():
    """
    Create a prompt message for a financial Bot assistant.

    This function generates a message for a financial Bot assistant designed to answer questions related to
    documents related to NASDAQ. The message includes instructions for the assistant, a context section that
    should be used to answer questions, and a template for presenting questions and agent's thoughts.

    The message template includes placeholders for the question, agent's scratchpad, and the provided context.

    Returns:
        message (SystemMessage): A prompt message for the financial Bot assistant.
    """

    message = SystemMessage(
        content=(
            """
        You are a financial Bot assistant for answering any questions related with the given documents related with NASDAQ.
        Please follow below instruction:
        Use the following context (delimited by <ctx></ctx>) to answer the question, do not rely on any prior knowledge. If you dont know the answer with the given
        context just say "I don't know".
        --------------------------
        <ctx>
        {context}
        </ctx>
        -------------------------
        If you can't answer the question with the above context then you have permisson to use Search tool.
    

        Question: {question}
        Thought:{agent_scratchpad}
        """
        )
    )
    return message


with st.sidebar:
    openaikey = st.text_input("OPENAI_API_KEY", type="password")
    os.environ["OPENAI_API_KEY"] = openaikey
    # LOAD TOKEN
    if st.button("Add Token"):
        tokenresponse = requests.get(f"http://money1:8000/load_token?token={openaikey}")

    st.divider()
    st.subheader("Your finance documents")
    uploaded_files = st.file_uploader(
        "Upload your PDFs here and click on 'Add Data'", accept_multiple_files=True
    )
    if st.button("Add Data") and openaikey:
        with st.spinner("Adding Data..."):
            documents = []
            url = "http://money1:8000/load_pdfs"
            data_list = []
            counter = 1
            for f in uploaded_files:
                data = ("files", f)
                data_list.append(data)
            st.write(len(data_list))
            response = requests.post(url, files=data_list)
            st.write(response.content)

    on = st.toggle("ACTIVATE GECKO")


if on:
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", max_tokens=1000)
    memory = AgentTokenBufferMemory(memory_key="history", llm=llm, max_token_limit=1000)

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=create_agent_prompt(),
        extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
    )

    agent = OpenAIFunctionsAgent(llm=llm, tools=create_agent_tools(), prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=create_agent_tools(),
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=3,
    )

    starter_message = "Ask me anything about finanance"
    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [AIMessage(content=starter_message)]

    for msg in st.session_state.messages:
        if isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        memory.chat_memory.add_message(msg)

    if prompt := st.chat_input(placeholder=starter_message):
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = agent_executor(
                {"input": prompt, "history": st.session_state.messages},
                callbacks=[st_callback],
                include_run_info=True,
            )
            st.session_state.messages.append(AIMessage(content=response["output"]))
            st.write(response["output"])
            memory.save_context({"input": prompt}, response)
            st.session_state["messages"] = memory.buffer
            run_id = response["__run"].run_id
else:
    st.header("Welcome to MoneyMaker! 💵")
    col1, col2 = st.columns(2)
    with col1:
        st.image("moneymaking.png", width=250)
    with col2:
        st.write(
            "<p>Welcome to <strong>MoneyMaker</strong> -- your answer to all your financial questions. In order to help you bear in mind the following instructions:</p>",
            unsafe_allow_html=True,
        )
        st.write(
            "<ol><li>Enter your Open AI key</li><li>Load your financial documents in PDF</li></ol>",
            unsafe_allow_html=True,
        )
        st.write(
            "<p>Oh and if you are brave give <strong>GECKO</strong> a try :)</p>",
            unsafe_allow_html=True,
        )
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message(message["role"], avatar=av_us):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"], avatar=av_ass):
                st.markdown(message["content"])

    # ACCEPT USER INPUT
    if myprompt := st.chat_input(
        "Greetings stranger, I am MoneyMaker a genius in finance, how can I help you?"
    ):
        # ADD CHAT HISTORY
        st.session_state.messages.append({"role": "user", "content": myprompt})

        # USER MESSAGES
        with st.chat_message("user", avatar=av_us):
            st.markdown(myprompt)
            usertext = f"user: {myprompt}"

        # CHATBOT MESSAGES
        with st.chat_message("assistant", avatar=av_ass):
            message_placeholder = st.empty()
            full_response = ""
            apiresponse = requests.get(f"http://money1:8000/model?question={myprompt}")
            risposta = apiresponse.content.decode("utf-8")
            res = risposta[1:-1]
            response = res.split(" ")
            # TYPING EFFECT
            for r in response:
                full_response = full_response + r + " "
                message_placeholder.markdown(full_response + "▌")
                sleep(0.1)
            message_placeholder.markdown(full_response)
            asstext = f"assistant: {full_response}"
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
