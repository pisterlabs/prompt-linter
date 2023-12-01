'''
PandasAI is a library that allows you to use the power of language models to interact with your data.
'''

#%% ---------------------------------------------  IMPORTS  ----------------------------------------------------------
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from pandasai.llm.open_assistant import OpenAssistant
from pandasai.llm.starcoder import Starcoder
from credentials import OPENAI_API_KEY
import os
import streamlit as st
import matplotlib.pyplot as plt
from langchain.callbacks import get_openai_callback
import tempfile

# --------------------  SETTINGS  -------------------- #
st.set_page_config(page_title="Home", layout="wide")
st.markdown("""<style>.reportview-container .main .block-container {max-width: 95%;}</style>""", unsafe_allow_html=True)

# --------------------- HOME PAGE -------------------- #
st.title("PANDAS AI (Chart Generator) 🐼")
st.write("""PandasAI is a library that allows you to use the power of language models to interact with your data. Add a file and start creating visuals and insights using only natural language.""")
st.write("Let's start interacting with PandasAI!")

# ------------------ SIDE BAR SETTINGS ------------------
st.sidebar.subheader("Settings:")
tts_enabled = st.sidebar.checkbox("Enable Text-to-Speech", value=False)
ner_enabled = st.sidebar.checkbox("Enable NER in Response", value=False)



# Add a feature for the user to input csv file
csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

#Add a feature for the user to input a question
question = st.text_input("Enter a question")

#%% ---------------------------------------------  MODELS  -----------------------------------------------------------#
# Define List of Models
models = {
    "OpenAI": OpenAI,
    "Starcoder": Starcoder,
    "Open-Assistant": OpenAssistant
}
#@title Select Model to Run
model_to_run = 'OpenAI'

# Enter API Key
API_KEY = OPENAI_API_KEY #@param {type:"string"}

# If question is not empty and csv file is not empty
if csv_file:


        df = pd.read_csv(csv_file)

        with st.expander("Document Expander (Press button on the right to fold to fold or unfold)", expanded=True):
            st.subheader("Uploaded Document:")
            st.dataframe(df.head(30))

        # Model Initialisation
        llm = models[model_to_run](api_token=API_KEY)
        pandas_ai = PandasAI(llm, conversational=False, verbose=True)

       
        if question:
            # Enter Prompt related to data or Select from Pre-defined for demo purposes.
            prompt = question
            with get_openai_callback() as cb:
                response = pandas_ai.run(df, prompt=prompt,
                          is_conversational_answer=False)

                # Show the response using stream lit but no st.write
                st.write(response)

                plt.plot()
                plt.xlabel("")
                plt.ylabel("")
                plt.title("")

                plt.gcf().autofmt_xdate()
                plt.gcf().auto_layout = True
                plt.gcf().tight_layout()
                plt.gcf().subplots_adjust(bottom=0.15)
                plt.gcf().subplots_adjust(left=0.15)

                st.pyplot(plt)
                st.write(cb)
                # Create a buton for the user to download the cb file
                st.download_button("Download PDF", data = response, file_name = "response.pdf")
                
else:
    st.write("Please upload a CSV file")

