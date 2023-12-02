import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

st.sidebar.success("Select an agent above.")
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="langchain_search_api_key_openai", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/2_Chat_with_search.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

#st.title("🔎 Competition Screener")

#st.set_page_config(page_title="Competition Screener", page_icon="🔎")

st.markdown("# SPEC for LP/AD")
st.sidebar.header("SPEC for LP/AD")
st.write(
    """Create specification to generate Landing Pages and Ads for micro-segments. Enjoy!"""
)



if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a Spec Agent agent who can create a specification of required elements for a Performance Ad and a Landing Page, so i can use it to generate them for micro-segments."}
    ]

if target_product := st.chat_input(placeholder="Write a target product, for example : fitness app"):
    st.session_state.messages.append({"role": "user", "content": target_product})
    st.chat_message("user").write(target_product)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # Define your prompt template here for Spec Agent
    spec_agent_prompt = f'''
    You are a Spec Agent with the role "SPEC for LP/AD".
    GOALS: Create a specification of required elements for a Performance Ad and a Landing Page, so i can use it to generate them for micro-segments

    constraints :
    - If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember
    - Ensure the tool and args are as per current plan and reasoning

    instruction : Define a List of required components for a performance Ad
    
    Using the information provided, create specifications to generate Landing Pages and Ads for the target product: {target_product}.
    '''

    st.session_state["messages"].append({"role": "assistant", "content": spec_agent_prompt})

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)