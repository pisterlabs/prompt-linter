import streamlit as st
import pandas as pd
import os


from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler

from langchain.llms import OpenAI

if "data_ready" not in st.session_state or not st.session_state.logged:
    st.warning("Bitte zuerst anmelden!")


else:
    df = st.session_state.kb_data_merged
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if "pandas_df_agent" not in st.session_state:
        st.session_state.pandas_df_agent = create_pandas_dataframe_agent(
            OpenAI(temperature=0), df, verbose=True
        )

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    if "messages" not in st.session_state or st.sidebar.button(
        "Clear conversation history"
    ):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="Wer macht die meisten Punkte?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = st.session_state.pandas_df_agent.run(
                st.session_state.messages, callbacks=[st_cb]
            )
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
