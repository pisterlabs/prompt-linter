import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All

load_dotenv()

# local_path = (
#     "/Users/mkozlowski/Library/Application Support/nomic.ai/GPT4All/wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin"
# )
local_path = (
    "/Users/mkozlowski/Library/Application Support/nomic.ai/GPT4All/ggml-model-gpt4all-falcon-q4_0.bin"
)
callbacks = [StreamingStdOutCallbackHandler()]


def main():
    st.set_page_config(page_title="Ask your CSV")
    st.title("Chat with CSV using Gpt4All 🤖🟥")

    uploaded_file = st.file_uploader("Upload your Data", type="csv")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        df = pd.read_csv(tmp_file_path)

        gpt_4_all_model = GPT4All(model=local_path, callbacks=callbacks, verbose=True, max_tokens=4096)
        PREFIX = """
                    You are working with a pandas dataframe in Python. You should answer exactly to the question posed of you. Expect some data to be incomplete or empty. The name of the dataframe is `df`. You should use the tools below to answer the question posed of you:"""
        agent_panda = create_pandas_dataframe_agent(gpt_4_all_model, df, verbose=True, prefix=PREFIX)

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(agent_panda.run(user_question))


if __name__ == "__main__":
    main()
