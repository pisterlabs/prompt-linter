"""
# My first app
Here's our first attempt at using data to create a table:
"""
import streamlit as st
import openai
from datetime import datetime
from streamlit.components.v1 import html
import pandas as pd
import csv
st.set_page_config(page_title="GPT Write")



input_text = None

st.markdown("""
# GPT Write
""")
input_text = st.text_input("Brainstorm ideas for", disabled=False, placeholder="What's on your mind?")

# st.markdown(hide, unsafe_allow_html=True)


st.markdown(
    """
    <style>
        iframe[width="220"] {
            position: fixed;
            bottom: 60px;
            right: 40px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
if input_text:
    prompt = "Brainstorm ideas for "+str(input_text)
    if prompt:
        openai.api_key = st.secrets['openaikey']
        response = openai.Completion.create(
          model="text-davinci-003",
          prompt=f"rewrite my message, correct the grammar and make it more friendly, natural, shorter, and clearer. {input_text}",
          temperature=1,
          max_tokens=100,
          top_p=1.0,
          frequency_penalty=0.57,
          presence_penalty=0.0
        )
        brainstorming_output = response['choices'][0]['text']
        
        today = datetime.today().strftime('%Y-%m-%d')
        topic = "Brainstorming ideas for: "+input_text+"\n@Date: "+str(today)+"\n"+brainstorming_output
        
        st.info(brainstorming_output)
#         filename = "brainstorming_"+str(today)+".txt"
#         btn = st.download_button(
#             label="Download txt",
#             data=topic,
#             file_name=filename
#         )
#         fields = [input_text, brainstorming_output, str(today)]
#         # read local csv file
#         r = pd.read_csv('./data/prompts.csv')
#         if len(fields)!=0:
#             with open('./data/prompts.csv', 'a', encoding='utf-8', newline='') as f:
#                 # write to csv file (append mode)
#                 writer = csv.writer(f, delimiter=',', lineterminator='\n')
#                 writer.writerow(fields)

        
        

    