import streamlit as st
import csv
from io import StringIO
import pandas as pd
import os
from zipfile import ZipFile
import base64
import re
import openai
import configparser
from gtts import gTTS
from google.cloud import texttospeech

#Page configuration
st.set_page_config(
    page_title="Admin",
    )

try:
    tab1, tab2, tab3, tab4 = st.tabs(['Upload/Download Files', 'Generate content', 'Feedback', 'File list'])

    with tab1:
        st.error("Access denied.")
            
    with tab2:
        st.error("Access denied.")

    with tab3:
        
        try:
            feedback_file = "/mount/src/spanish-app/Feedback/reports.txt"

            with open(feedback_file, 'r') as f:
                reports = f.read()
                download_button = st.download_button(
                    label = "Download report",
                    data = reports,
                    file_name = "reports.txt",
                    mime = 'text/plain'
                )

            st.write(reports)

        except:
            st.error("No file found.")

    with tab4:
        st.error("Access denied.")
except:
    st.error("Access denied.")


#Show files in specified directory
#dir_list = os.listdir("/mount/src/spanish-app/Feedback")
#st.write(dir_list)

#Show current working directory - /mount/src/spanish-app
#cwd = os.getcwd()
#st.write(cwd)

#Show full directory for script - /mount/src/spanish-app/pages
#directory = os.path.dirname(os.path.abspath(__file__))
#st.write(directory)