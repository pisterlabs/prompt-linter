"""
Streamlit frontend
"""

### LIBRARIES

import boto3
import streamlit as st
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms import Bedrock
from langchain.retrievers import AmazonKendraRetriever

from utils.ask_human import CustomAskHumanTool
from utils.model_params import get_model_params
from utils.prompts import create_agent_prompt, create_qa_prompt

### PAGE ELEMENTS

st.set_page_config(
    page_title="RAG Agent Demo",
    page_icon="🦜",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.markdown("### Leveraging the User to Improve Agents in RAG Use Cases")

mode = st.selectbox(
    label="Select agent type",
    options=("Agent with AskHuman tool", "Traditional RAG Agent"),
)


### PARAMETERS

# model params
MODEL_REGION = "us-east-1"  # Bedrock region
MODEL_ID = "anthropic.claude-instant-v1"  # LLM to use
PARAMS = {
    "answer_length": 500,  # max number of tokens in the answer
    "temperature": 0.0,  # temperature during inference
    "top_p": 0.5,  # cumulative probability of sampled tokens
    "stop_words": [
        "\n\nHuman:",
    ],  # words after which the generation is stopped
}

# retriever params
KENDRA_INDEX_ID = "<YOUR-KENDRA-INDEX-ID>"
TOP_K = 3

# memory
MEMORY_SIZE = 3


### SET UP RETRIEVAL CHAIN

# retriever
retriever = AmazonKendraRetriever(index_id=KENDRA_INDEX_ID, top_k=TOP_K)

# Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=MODEL_REGION,
)

# LLM
model_params = get_model_params(model_id=MODEL_ID, params=PARAMS)
llm = Bedrock(
    client=bedrock_client,
    model_id=MODEL_ID,
    model_kwargs=model_params,
)

# QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": create_qa_prompt(),
    },
)


### SET UP TOOLS & AGENT

# memory
conversational_memory = ConversationBufferMemory(
    memory_key="chat_history", k=MEMORY_SIZE, return_messages=True
)

# tool for Kendra search
kendra_tool = Tool(
    name="KendraRetrievalTool",
    func=qa_chain,
    description="""Use this tool first to answer human questions. The input to this tool should be the question.""",
)

# tool for asking human
human_ask_tool = CustomAskHumanTool()

# agent prompt
prefix, format_instructions, suffix = create_agent_prompt()

# initialize agent
agent = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=[human_ask_tool, kendra_tool]
    if mode == "Agent with AskHuman tool"
    else [kendra_tool],
    llm=llm,
    verbose=True,
    max_iterations=5,
    early_stopping_method="generate",
    memory=conversational_memory,
    agent_kwargs={
        "prefix": prefix,
        "format_instructions": format_instructions,
        "suffix": suffix,
    },
)

# question form
with st.form(key="form"):
    user_input = st.text_input("Ask your question")
    submit_clicked = st.form_submit_button("Submit Question")

# output container
output_container = st.empty()
if submit_clicked:
    output_container = output_container.container()
    output_container.chat_message("user").write(user_input)

    answer_container = output_container.chat_message("assistant", avatar="🦜")
    st_callback = StreamlitCallbackHandler(answer_container)

    answer = agent.run(user_input, callbacks=[st_callback])

    answer_container = output_container.container()
    answer_container.chat_message("assistant").write(answer)
