import streamlit as st
st.set_page_config(page_title="LibertyGPT Sandbox", layout='wide')
st.title('LibertyGPT Sandbox')
from llama_index import LLMPredictor, ServiceContext, StorageContext, load_index_from_storage
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents import AgentType, Tool, initialize_agent, load_tools
from langchain.chains import LLMMathChain
from langchain.callbacks import StreamlitCallbackHandler
from llama_index.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool
from llama_index.langchain_helpers.agents import create_llama_chat_agent, create_llama_agent 
from llama_index import LangchainEmbedding
from llama_index.query_engine import RetrieverQueryEngine


ho3_directory = "../_policy_index_metadatas"
doi_directory = "../_index_storage"
uniform_building_codes = "../_property_index_storage"


st.write("This sandbox is powered by :statue_of_liberty:**GPT**, 🦜[LangChain](https://langchain-langchain.vercel.app/docs/get_started/introduction.html) and :llama:[Llama-Index](https://gpt-index.readthedocs.io/en/latest/index.html)", 
          unsafe_allow_html=True)


def get_llm(temperature=0):
    return ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")


def get_llm_predictor(temperature=0):
    return LLMPredictor(ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo"))


def get_embed_model():
    return LangchainEmbedding(OpenAIEmbeddings())


def initialize_index(storage_directory):
    llm = get_llm()
    embed_model = get_embed_model()

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        )

    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=storage_directory),
        service_context=service_context,
    )
    return index

ho3_index = initialize_index(storage_directory=ho3_directory)
doi_index = initialize_index(storage_directory=doi_directory)
bldg_code_index = initialize_index(storage_directory=uniform_building_codes)

llm_math_chain = LLMMathChain.from_llm(llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"))

tools = [
    Tool(
        name="ho3_query_engine",
        func=lambda q: str(ho3_index.as_query_engine(
            similarity_top_k=10,
            streaming=False).query(q)),
        description="useful for when you want to answer questions about homeowner's insurance coverage.",
        return_direct=False,
    ),
    Tool(
        name="doi_query_engine",
        func=lambda q: str(doi_index.as_query_engine(
            similarity_top_k=10,
            streaming=False).query(q)),
        description="useful for when you want to answer questions department of insurance (DOI) regulation such as rules, requirements, or statutes.",
        return_direct=False,
    ),
        Tool(
        name="bldg_codes_query_engine",
        func=lambda q: str(bldg_code_index.as_query_engine(
            similarity_top_k=10,
            streaming=False).query(q)),
        description="useful for when you want to answer questions about building codes.",
        return_direct=False,
    ),
        Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),    
]


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", streaming=False)

agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True
)


question_container = st.empty()
results_container = st.empty()
sources_container = st.empty()

res = results_container.container()
source = sources_container.container()

if prompt := st.text_input(label="Send a message"):
    st.chat_message("user", avatar="https://raw.githubusercontent.com/pdoubleg/junk-drawer/main/src_index/data/icons/c3po_icon_resized_pil.jpg").write(prompt)
    with st.chat_message("assistant", avatar="https://raw.githubusercontent.com/pdoubleg/junk-drawer/main/src_index/data/icons/user_question_resized_pil.jpg"):
        st_callback = StreamlitCallbackHandler(
        parent_container=res,
        max_thought_containers=5,
        expand_new_thoughts=True,
        collapse_completed_thoughts=False,
    )
        response = agent.run(prompt, callbacks=[st_callback])
        question_container.write(f"**Question:** {prompt}")
        res.write(f"**Answer:** {response}")

# cols2 = st.columns(2, gap="small")
# with cols2[0]:
#     question_container = st.empty()
#     results_container = st.empty()
    
# with cols2[1]:
#     sources_container = st.empty()

# # A hack to "clear" the previous result when submitting a new prompt.
# from callbacks.clear_results import with_clear_container

# if with_clear_container(submit_clicked):
#     # Create our StreamlitCallbackHandler
#     res = results_container.container()
#     source = sources_container.container()
#     streamlit_handler = StreamlitCallbackHandler(
#         parent_container=res,
#         max_thought_containers=int(max_thought_containers),
#         expand_new_thoughts=expand_new_thoughts,
#         collapse_completed_thoughts=collapse_completed_thoughts,)

    # question_container.write(f"**Question:** {mrkl_input}")

    # answer = mrkl.run(mrkl_input, callbacks=[streamlit_handler])
    # res.write(f"**Answer:** {answer}")
    # source.write("hello")

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
