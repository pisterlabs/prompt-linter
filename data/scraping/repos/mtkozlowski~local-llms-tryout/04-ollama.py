import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_pandas_dataframe_agent, create_csv_agent
from langchain.llms import Ollama

load_dotenv()


def main():
    st.set_page_config(page_title="Ask your CSV")
    st.title("Chat with CSV using Ollama 🦙")

    uploaded_file = st.file_uploader("Upload your Data", type="csv")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        df = pd.read_csv(tmp_file_path)
        ollama_model = Ollama(model="phind-codellama", temperature=0.3, verbose=True)

        PREFIX = """
        You are working with a pandas dataframe in Python. You should answer exactly to the question posed of you. The name of the dataframe is `df`.
        You should use the tools below to answer the question posed of you:"""

        SUFFIX_WITH_DF = """
        This is the result of `print(df.head())`:
        {df_head}

        Begin!
        Question: {input}
        {agent_scratchpad}"""

        agent_panda = create_pandas_dataframe_agent(
            ollama_model, df, verbose=True, prefix=PREFIX)

        agent = create_csv_agent(
            Ollama(model="phind-codellama", temperature=0.3, verbose=True),
            tmp_file_path,
            verbose=True,
            prefix=PREFIX
        )

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(agent.run(user_question))


if __name__ == "__main__":
    main()
