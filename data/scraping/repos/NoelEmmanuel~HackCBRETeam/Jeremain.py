import streamlit as st
from login import login_page
from insight import insight_page
import os
import openai
import csv
import myGPT

# Define the roles list
roles = ["Technician", "Building Manager", "Account Director", "Chief Engineer", "Portfolio Manager", "Asset Manager", "Leasing Manager", "Facility Coordinator", "Maintenance Supervisor"]

# Create a Streamlit app title
st.title("Multi-Page Streamlit App")

# Create a session state to manage page switching
if 'page' not in st.session_state:
    st.session_state.page = 'login'

# Check the page state and render the appropriate page
if st.session_state.page == 'login':
    login_page()  # Pass the roles list to the login_page function
else:
    insight_page()