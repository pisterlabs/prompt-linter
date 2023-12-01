import streamlit as st
from io import StringIO
import pandas as pd
import time

from openai_connection import get_openai_ans

num_cols = 2
max_desc_chars = 500
max_question_chars = 100
delay_time = 2

def get_file_input():
    return

def run_demo():
    st.title("OpenAI Green Skills & Jobs classifier.")
    col1, col2 = st.columns(num_cols)    
    with col1:
        job_descp = st.text_area(label="Job description", placeholder="Please input a job description", max_chars=int(max_desc_chars))
        do_analysis = st.button(label="Analyse")
        questions = st.text_area(label="Ask your questions here:", placeholder="Is this a green job?", max_chars=int(max_question_chars))
        do_answers = st.button(label="Answer questions")
        with st.expander("Input a CSV file (for debug purposes):"):
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                # To read file as bytes:
                bytes_data = uploaded_file.getvalue()
                st.write(bytes_data)

                # To convert to a string based IO:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                st.write(stringio)

                # To read file as string:
                string_data = stringio.read()
                st.write(string_data)

                # Can be used wherever a "file-like" object is accepted:
                dataframe = pd.read_csv(uploaded_file)
                st.write(dataframe)
    with col2:
        st.write("Results")
        out = ""
        if do_answers:
            with st.spinner("Analysis in progress..."):
                answers = get_openai_ans(job_descp + questions)
                time.sleep(delay_time)
            out += answers["text"]
        job_insights = st.markdown(out)

if __name__=="__main__":
    run_demo()