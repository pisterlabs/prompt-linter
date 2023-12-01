import os
import json
import dotenv
import openai
import json
import pandas as pd
import streamlit as st

# .env file must have OPENAI_API_KEY and OPENAI_API_BASE
dotenv.load_dotenv()
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")
ENGINE = os.environ.get("ENGINE")
TEMPERATURE = 0.0
MAX_TOKENS = 500
TOP_P = 0.0
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0

def load_contracts():
    #read two text files
    return open("./contract/version1.txt", "r").read(), open("./contract/version2.txt", "r").read()

# define custom function to run the openai.ChatCompletion.create function
def run(user_msg: str, system_msg: str, engine: str = ENGINE):  
    """
    This function runs the openai.ChatCompletion.create function
    """
    messages = [{"role":"system", "content":system_msg},
                {"role":"user","content":user_msg}
                ]

    res = openai.ChatCompletion.create(
        engine=engine,
        messages = messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
        n=1
        )
    
    return res.choices[0].message['content']

def DocumentComparison():
    """
    This function runs the openai.ChatCompletion.create function
    """

    system_msg = """
You are a contract reviewer in a legal team. You are responsible for reviewing the contract and providing insights. When answering a question, start with a simple conclusion, and then explain in detail with quotes from the contract.

## Process 

Read a line from each document and compare.
If there is no difference, then move to the next line.
If there is a difference remember the line and the differences
Repeats end of the document.

Make a markdown table to show the difference that is found during the process.
The table includes Line, Versions, and the difference in bold

## Response Example
[Use a table to summarize the answers]

There is two differences.

|Item Number|Line|Verion 1|Version2|
|-|-|-|-|
|1|10|**1234**|**5678**|
|2|12|**ABCD**|**EFGH**|


## Safety
Use only the given documents. Do not use any other documents.
"""
    user_msg = ""


    # sidebar
    with st.sidebar:
        with st.container():
            st.info("Compare two documents. Type a question and click __Ask__ button to get answer.")
        sample_tab, system_tab = st.tabs(["samples", "system"])
        with sample_tab:
            st.write("## Sample Questions")
            st.code("what is the risk for the employee with the new updated version", language="html")
        with system_tab:
            st.write("## System Message")
            st.text_area(label="System", value=system_msg, height=800)

        
    # load documents
    ver1, ver2 = load_contracts()

    st.markdown("# Compare two documents")
    st.markdown("This demo will show you how to use Azure OpenAI to compare two documents")
    with st.expander("Demo scenario"):
        st.image("https://github.com/hyssh/azure-openai-quickstart/blob/main/images/Architecture-demo-docs.png?raw=true")
        st.markdown("1. User will type text (multi lines) in the input box")
        st.markdown("2. __Web App__ sends the text to __Azure OpenAI__")
        st.markdown("3. __Azure OpenAI__ classify the text line by line and return the classification results in JSON format")
        st.markdown("4. __Web App__ shows the results on screen and user can download the results as CVS file")
    
    st.markdown("---")
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            with st.container():
                st.write("### version 1")
                v1_text_area = st.text_area(label="contents", value=ver1, height=500)
        with col2:
            with st.container():
                st.write("### version 2")
                v2_text_area = st.text_area(label="contents", value = ver2, height=500)

    st.markdown("---")

    with st.container():
        selected_engine = st.selectbox("Select a GPT model", ["chat-gpt", "gpt4"], index=0)
        user_questions = st.text_input("Ask a question")
        if st.button("Ask"):
            st.spinner("Comparing...")
            user_msg = f"There are two dcouments. version 1 --- {v1_text_area} --- version 2 --- {v2_text_area} ---. Please compare the two documents and answer my question {user_questions}"
            with st.container():
                st.info(run(user_msg, system_msg, selected_engine))
        else:
            with st.container():
                st.empty()


if __name__ == "__main__":
    DocumentComparison()    