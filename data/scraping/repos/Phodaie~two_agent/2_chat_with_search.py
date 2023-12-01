
import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from schema import LlmModelType, get_completion_from_messages
from schema import TwoAgentsSettings 

with st.sidebar:
    model_names = [enum.value for enum in LlmModelType]
    model_name = st.selectbox('Model', model_names, 0)
    selected_model = LlmModelType(model_name)

st.title("Chat with search")


if "messages" not in st.session_state:
  
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()

    llm = ChatOpenAI(model_name=selected_model.value, streaming=True)
    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)