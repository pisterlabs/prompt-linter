from dotenv import load_dotenv
import os
import BackEndRoutes.streamlit as st
import pandas as pd
from pandasai import PandasAI
import matplotlib.pyplot as plt



from langchain.llms import GooglePalm

llm = GooglePalm(
    model='models/text-bison-001',
    temperature=0,
    max_output_tokens=8196,
    google_api_key='AIzaSyA1fu-ob27CzsJozdr6pHd96t5ziaD87wM'
)

# llm = OpenAssistant(api_token=API_KEY)
# llm = Falcon(api_token=API_KEY)

# to disable pandasai.log: PandasAI(llm, enable_logging=False)
pandas_ai = PandasAI(llm)


st.title('Analysis CSV file with Ai')

uploaded_file = st.file_uploader('Upload your CSV file', type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head(3))

    prompt = st.text_input('Enter your prompt')

    if st.button('Generate'):
        if prompt:
            with st.spinner("generating response..."):
                output = pandas_ai.run(df, prompt=prompt)

                # Convert Axes object to Figure object
                if isinstance(output, plt.Axes):
                    fig = output.figure
                    st.pyplot(fig)
                else:
                    st.write(output)
        else:
            st.warning("Please enter your prompt.")