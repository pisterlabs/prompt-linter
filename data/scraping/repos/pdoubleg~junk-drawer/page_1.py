from pathlib import Path

import streamlit as st

st.set_page_config(page_title="LibertyGPT Sandbox", layout='wide')
st.title('GPT Sandbox:  🦜MRKL')

"""
This Streamlit app showcases a LangChain agent that replicates the "Modular Reasoning, Knowledge and Language system", aka the
[MRKL chain](https://arxiv.org/abs/2205.00445).

"""

from langchain.agents import AgentType, initialize_agent, Tool
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import StreamlitCallbackHandler

from llama_index import ServiceContext, StorageContext, LangchainEmbedding, load_index_from_storage
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, Tool, initialize_agent, load_tools
    
ho3_directory = "../_policy_index_metadatas"
doi_directory = "../_index_storage"
uniform_building_codes = "../_property_index_storage"   
    
    
def get_llm(temperature=0):
    return ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")


def get_embed_model():
    return LangchainEmbedding(OpenAIEmbeddings())


@st.cache_resource
def initialize_index(storage_directory):
    llm = get_llm()
    embed_model=get_embed_model()

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model)

    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=storage_directory),
        service_context=service_context,
    )
    return index


ho3_index = initialize_index(storage_directory=ho3_directory)
doi_index = initialize_index(storage_directory=doi_directory)
bldg_code_index = initialize_index(storage_directory=uniform_building_codes)


tools = [
    Tool(
        name="ho3_query_engine",
        func=lambda q: str(ho3_index.as_query_engine(
            similarity_top_k=5,
            streaming=False).query(q)),
        description="useful for when you want to answer questions about homeowner's insurance coverage.",
        return_direct=False,
    ),
    Tool(
        name="doi_query_engine",
        func=lambda q: str(doi_index.as_query_engine(
            similarity_top_k=5,
            streaming=False).query(q)),
        description="useful for when you want to answer questions about Department of Insurancce (DOI) regulations such as rules, statutes, or general requirements insurance companies must follow.",
        return_direct=False,
    ),
        Tool(
        name="bldg_codes_query_engine",
        func=lambda q: str(bldg_code_index.as_query_engine(
            similarity_top_k=5,
            streaming=False).query(q)),
        description="useful for when you want to answer questions about building consruction, and renovation.",
        return_direct=False,
    ),
]


mrkl = initialize_agent(
        tools, 
        llm=OpenAI(temperature=0, streaming=True), 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True
    )


expand_new_thoughts = st.sidebar.checkbox(
    "Expand New Thoughts",
    value=True,
    help="True if LLM thoughts should be expanded by default",
)

collapse_completed_thoughts = st.sidebar.checkbox(
    "Collapse Completed Thoughts",
    value=True,
    help="True if LLM thoughts should be collapsed when they complete",
)


max_thought_containers = st.sidebar.number_input(
    "Max Thought Containers",
    value=5,
    min_value=1,
    help="Max number of completed thoughts to show. When exceeded, older thoughts will be moved into a 'History' expander.",
)


SAVED_SESSIONS = {
    "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?": "leo.pickle",
    "What is the full name of the artist who recently released an album called "
    "'The Storm Before the Calm' and are they in the FooBar database? If so, what albums of theirs "
    "are in the FooBar database?": "alanis.pickle",
}


key = "input"
shadow_key = "_input"

if key in st.session_state and shadow_key not in st.session_state:
    st.session_state[shadow_key] = st.session_state[key]

with st.form(key="form"):

    mrkl_input = st.text_input("Ask a question", key=shadow_key)
    st.session_state[key] = mrkl_input
    submit_clicked = st.form_submit_button("Submit Question")

cols2 = st.columns(2, gap="small")
with cols2[0]:
    question_container = st.empty()
    results_container = st.empty()
    
with cols2[1]:
    sources_container = st.empty()

# A hack to "clear" the previous result when submitting a new prompt.
from callbacks.clear_results import with_clear_container

if with_clear_container(submit_clicked):
    # Create our StreamlitCallbackHandler
    res = results_container.container()
    source = sources_container.container()
    streamlit_handler = StreamlitCallbackHandler(
        parent_container=res,
        max_thought_containers=int(max_thought_containers),
        expand_new_thoughts=expand_new_thoughts,
        collapse_completed_thoughts=collapse_completed_thoughts,
    )

    question_container.write(f"**Question:** {mrkl_input}")

    answer = mrkl.run(mrkl_input, callbacks=[streamlit_handler])
    res.write(f"**Answer:** {answer}")
    source.write("hello")