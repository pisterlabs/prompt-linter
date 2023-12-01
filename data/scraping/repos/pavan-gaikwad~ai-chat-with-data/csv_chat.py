
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI, GooglePalm
import streamlit as st
import csv
from langchain.memory import ConversationBufferMemory



def main():
    st.set_page_config(page_title="AI Data Analyst")
    st.header("AI Data Analyst")

    openai_api_key = st.text_input("OpenAI API Key", type="password")
    csv_file = st.file_uploader("Upload a CSV file", type=["csv"])
    memory = ConversationBufferMemory(memory_key="chat_history")
    if csv_file is not None:
        # save csv file to disk at location /csvfiles
        filename = f"csvfiles/{csv_file.name}"
        with open(filename, "wb") as f:
            f.write(csv_file.getbuffer())
        
        llm = OpenAI(temperature=0,
                    openai_api_key= openai_api_key)
        user_query = st.text_input("What is your question?")
        if user_query:
            agent = create_csv_agent(path=filename, llm=llm, verbose=True, memory=memory)
            response = agent.run(user_query)
            st.write(response)


if __name__ == "__main__":
    main()


