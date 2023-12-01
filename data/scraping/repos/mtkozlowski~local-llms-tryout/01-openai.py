import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_pandas_dataframe_agent, create_csv_agent, AgentType
from langchain.chat_models import ChatOpenAI

load_dotenv()


def main():
    st.set_page_config(page_title="Ask your CSV")
    st.title("Chat with CSV using OpenAI 🤖")

    uploaded_file = st.file_uploader("Upload your Data", type="csv")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        df = pd.read_csv(tmp_file_path)

        PREFIX = """
                You are working with a pandas dataframe in Python. Expect some data to be incomplete or empty. The name of the dataframe is `df`. You should use the tools below to answer the question posed of you:"""

        open_ai_chat_model = ChatOpenAI()

        agent = create_csv_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
            tmp_file_path,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            prefix=PREFIX
        )

        agent_panda = create_pandas_dataframe_agent(open_ai_chat_model, df, verbose=True, prefix=PREFIX)

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(agent.run(user_question))


if __name__ == "__main__":
    main()
