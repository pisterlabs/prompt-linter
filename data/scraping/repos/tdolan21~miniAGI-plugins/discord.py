import pandas as pd
import os
from langchain.document_loaders.discord import DiscordChatLoader
import streamlit as st
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI



tools = []

model = ChatOpenAI(temperature=0)
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)


st.title("Chat with your discord data")
st.subheader("")

path = ("documents/discord_data")
li = []
for f in os.listdir(path):
    expected_csv_path = os.path.join(path, f, "messages.csv")
    csv_exists = os.path.isfile(expected_csv_path)
    if csv_exists:
        df = pd.read_csv(expected_csv_path, index_col=None, header=0)
        li.append(df)

df = pd.concat(li, axis=0, ignore_index=True, sort=False)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)