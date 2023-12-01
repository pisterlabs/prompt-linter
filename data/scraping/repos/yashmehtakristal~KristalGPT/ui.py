# All imports

import streamlit as st
import openai
import os
from streamlit.logger import get_logger
logger = get_logger(__name__)
from typing import List
from typing import NoReturn


## Importing functions

# Function to check if question is entered
def is_query_valid(query: str) -> bool:
    if not query:
        st.error("Please enter a question!")
        return False
    return True

# Function to handle errors in reading the file
def display_file_read_error(e: Exception, file_name: str) -> NoReturn:
    
    st.error("Error reading file. Make sure the file is not corrupted or encrypted")

    # {Log the "type of exception occured}: {error message}. Extension: {extension of file}"
    logger.error(f"{e.__class__.__name__}: {e}. Extension: {file_name.split('.')[-1]}")

    # Stop execution
    st.stop()




