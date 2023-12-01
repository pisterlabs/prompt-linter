import pandas as pd
import streamlit as st
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI


st.set_page_config(
        page_title="Hello",
        page_icon="👋",
        )

st.write('# Welcome to ChatData! 👋')
st.subheader('Chat anything with your Data!')

st.markdown(
    """
     **This app using GPT Model,** so make sure don't upload any confidential data here.
    """
    )

OPENAI_API_KEY = st.text_input(label="Add Your OPENAI API KEY", value="")
st.markdown("If you don't know how to get an OPEN API Key. [Check this blog!](https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/).")

if OPENAI_API_KEY != "":
    llm = OpenAI(api_token=OPENAI_API_KEY)
    pandas_ai = PandasAI(llm)

    st.subheader('Upload your Data')
    file_upload = st.file_uploader(label="Choose a CSV file")


    if file_upload is not None:
        st.subheader('Sample Data')
        data = pd.read_csv(file_upload)
        st.dataframe(data.sample(10))
        st.subheader('Question')
        question = st.text_input(label="Add questions to your data", value="")
        if question != "":
            st.subheader('Result:')
            st.write(pandas_ai.run(data, prompt=question))