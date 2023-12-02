from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI as LangchainOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from streamlit_chat import message
import streamlit as st
import tempfile
import pandas as pd
from dotenv import load_dotenv

def answer_question(question, data):
    # Example logic to handle specific CSV data questions
    if "average" in question and "column" in question:
        column_name = question.split("column ")[1].strip()
        if column_name in data.columns:
            avg = data[column_name].mean()
            return f"The average value of column {column_name} is {avg}."

    # If no specific CSV logic is matched, return None
    return None

def main():
    load_dotenv()
    st.set_page_config(layout='wide', page_title="CSVChat powered by OpenAI and created by Srikanth Samy 🤖")
    st.title("CSVChat powered by OpenAI 🤖")
    st.subheader("Created by Srikanth Samy")

    input_csv = st.file_uploader("Upload your CSV file", type=['csv'])

    if 'past' not in st.session_state:
        st.session_state['past'] = []
        
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if input_csv is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(input_csv.getvalue())
            temp_path = temp_file.name

        st.info("CSV Uploaded Successfully")
        data = pd.read_csv(input_csv)
        st.dataframe(data, use_container_width=True)

        memory = ConversationBufferMemory(memory_key="chat_history")
        llm = LangchainOpenAI(temperature=1)
        template = """You are a nice chatbot having a conversation with a human.
                    Previous conversation:
                    {chat_history}
                    New human question: {question}
                    Response:"""
        prompt = PromptTemplate.from_template(template)
        conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)

        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to your CSV data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            csv_response = answer_question(user_input, data)
            if csv_response:
                response = csv_response
            else:
                output = conversation({"question": user_input})
                response = output['text']
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(response)

        if st.session_state['generated']:
            response_container = st.container()
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

if __name__ == "__main__":
    main()
