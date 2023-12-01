import streamlit as st
from streamlit_chat import message

import pandas as pd
from pandasai import PandasAI
import numpy as np

from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent


import os
# Retrieve the API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Instantiate a LLM
from pandasai.llm.openai import OpenAI
llm = OpenAI(api_token="YOUR_API_TOKEN")

# Initialize PandasAI
pandas_ai = PandasAI(llm)

# set page to wide mode
st.set_page_config(layout="wide")


# Check if DataFrame is already in session_state
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['Classification', 'Space Type', 'Room Name', 'Level', 'Room Count', 'Unit Count', 'NSF/Unit', 'NSF', 'Net to Gross Factor', 'GSF', 'Floor Finish', 'Wall Finish', 'Ceiling Finish'])

with st.sidebar:
    st.subheader('Instructions')
    st.info(
        """
        - You can start by uploading a CSV file or start from scratch.
        - Complete the input fields and click "Add to Table" to add data to the table.
        - To delete a row, enter the row index and click "Delete Row". The index is zero-based, i.e., the first row is index 0.
        - You can clear the entire data using the "Clear Data" button.
        - Finally, you can save your data as a CSV file with a filename of your choice.
        """
    )

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

    # Clear data button
    clear_data = st.button('Clear Data')

    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)

    if clear_data:
        st.session_state.df = pd.DataFrame(columns=['Classification', 'Space Type', 'Room Name', 'Level', 'Room Count', 'Unit Count', 'NSF/Unit', 'NSF', 'Net to Gross Factor', 'GSF', 'Floor Finish', 'Wall Finish', 'Ceiling Finish'])
        st.success('Data cleared.')

    # Delete row
    row_index = st.number_input('Enter row index to delete', value=-1, min_value=-1)
    delete = st.button('Delete Row')

    if delete:
        if row_index >= 0 and row_index < len(st.session_state.df):
            st.session_state.df = st.session_state.df.drop(st.session_state.df.index[row_index])
            st.session_state.df.reset_index(drop=True, inplace=True)
            st.success(f'Row {row_index} deleted.')

# Input fields in sidebar
with st.sidebar:
    st.subheader('Input Fields')
    classification = st.text_input('Classification', value='Revenue & Fan Experience')
    space_type = st.text_input('Space Type', value='Public Space')
    room_name = st.text_input('Room Name', value='Concourse')
    level = st.text_input('Level', value='Level 1')
    room_count = st.number_input('Room Count', value=1, format="%i")
    unit_count = st.number_input('Unit Count', value=1, format="%i")
    nsf_per_unit = st.number_input('NSF/Unit', value=1, format="%i")
    net_to_gross_factor = st.number_input('Net to Gross Factor', value=1.0)
    floor_finish = st.selectbox('Floor Finish', options=list(range(1, 6)))
    wall_finish = st.selectbox('Wall Finish', options=list(range(1, 6)))
    ceiling_finish = st.selectbox('Ceiling Finish', options=list(range(1, 6)))

    add_row = st.button('Add to Table')

if add_row:
    gsf_value = room_count * unit_count * nsf_per_unit * net_to_gross_factor
    df_new = pd.DataFrame({
        'Classification': [classification],
        'Space Type': [space_type],
        'Room Name': [room_name],
        'Level': [level],
        'Room Count': [room_count],
        'Unit Count': [unit_count],
        'NSF/Unit': [nsf_per_unit],
        'NSF': [room_count * unit_count * nsf_per_unit],
        'Net to Gross Factor': [net_to_gross_factor],
        'GSF': [np.round(gsf_value, 0)],  # rounding the GSF value
        'Floor Finish': [floor_finish],
        'Wall Finish': [wall_finish],
        'Ceiling Finish': [ceiling_finish]
    })
    st.session_state.df = pd.concat([st.session_state.df, df_new], axis=0)
    st.session_state.df.reset_index(drop=True, inplace=True)
    st.markdown(f"**Total GSF:** {np.round(st.session_state.df['GSF'].sum(), 0)}")  # rounding the total GSF value

# Display the DataFrame
st.dataframe(st.session_state.df)

# Save DataFrame as CSV
file_name = st.text_input('Enter filename to save as CSV')
if st.button('Save DataFrame as CSV') and file_name:
    st.session_state.df.to_csv(f'{file_name}.csv', index=False)
    st.success(f'DataFrame saved as {file_name}.csv')
