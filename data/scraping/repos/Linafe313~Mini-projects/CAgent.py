import streamlit as st
from langchain.llms import OpenAI
from langchain.agents import create_csv_agent
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
from dotenv import dotenv_values



def main():
    env_vars = dotenv_values()
    print(env_vars)
    st.set_page_config(page_title="Ask your CSV", page_icon=":robot:", layout="wide")
    st.title("Ask your CSV")

    user_csv = st.file_uploader("Upload your CSV file", type="csv")

    if user_csv is not None:
        user_question = st.text_input("Ask a question about your CVS: ")

        llm = OpenAI(temperature=0, max_tokens=1000, api_key=env_vars["OPENAI_KEY"])
        agent = create_csv_agent(llm, user_csv, verbose=True)

        if user_question is not None and user_question != "":
            response = agent.run(user_question)
            st.write(response)


if __name__ == "__main__":
    main()