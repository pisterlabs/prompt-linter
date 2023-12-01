import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain import HuggingFaceHub
from langchain.agents import create_pandas_dataframe_agent

# model = "tiiuae/falcon-40b"
model = "codellama/CodeLlama-34b-Python-hf"

load_dotenv()


def main():
    st.set_page_config(page_title="Ask your CSV")
    st.title("Chat with CSV using Falcon 🐦‍⬛")

    uploaded_file = st.file_uploader("Upload your Data", type="csv")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        df = pd.read_csv(tmp_file_path)

        falcon_model = HuggingFaceHub(
            repo_id=model, model_kwargs={"temperature": 0.3}
        )

        PREFIX = """
            You are working with a pandas dataframe in Python. You should answer exactly to the question posed of you. Expect some data to be incomplete or empty. The name of the dataframe is `df`. You should use the tools below to answer the question posed of you:"""

        agent_panda = create_pandas_dataframe_agent(falcon_model, df, verbose=True, prefix=PREFIX)

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(agent_panda.run(user_question))


if __name__ == "__main__":
    main()
