import streamlit as st
from concurrent.futures import as_completed
import streamlit as st
from streamlit_extras.app_logo import add_logo
import time
from langchain.callbacks import get_openai_callback

import sys, os
workingDirectory = os.getcwd()
dataDirectory = os.path.join(workingDirectory, "data")
chromaDirectory = os.path.join(workingDirectory, "ChromaDB")
analysisDirectory = os.path.join(workingDirectory, "Analysis")
miscellaneousDirectory = os.path.join(workingDirectory, "Miscellaneous")
costDirectory = os.path.join(workingDirectory, "cost_breakdown")
dirs = [
    workingDirectory,
    dataDirectory,
    chromaDirectory,
    analysisDirectory,
    miscellaneousDirectory,
    costDirectory
]
for d in dirs:
    if d not in sys.path: sys.path.append(d)

import chromaUtils
from ingestPdf import copyCollection
from Individual_Analysis import ind_analysis_main, get_yes_pdf_filenames
from Aggregated_Analysis import agg_analysis_main
from User_Input_Cleaning import process_user_input
from update_cost import update_usage_logs, Stage

from os import listdir
from os.path import abspath
from os.path import isdir
from os.path import join
from shutil import rmtree
import asyncio

st.set_page_config(layout="wide")
add_logo("images/temp_logo.png", height=100)

st.markdown("<h1 style='text-align: left; color: Black;'>PDF Filtering</h1>", unsafe_allow_html=True)
st.markdown('#')

async def my_async_function():
    await asyncio.sleep(2)  # Asynchronously sleep for 2 seconds

# Initialize session states (for purpose of page display and storing variables)
if 'pdf_filtered' not in st.session_state:
    st.session_state.pdf_filtered = False
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'pdf_filtering_time' not in st.session_state:
    st.session_state.pdf_filtering_time = None 

# collection_name, file_upload, prompt error messages
err_messages = {
    "000": "Please select an input collection to use, enter a research prompt and, enter a collection name (that hasn't been used yet) to store the filtered articles",
    "100": "Please enter a research prompt and a collection name (that hasn't been used yet) to store the filtered articles",
    "010": "Please select an input collection to use and a collection name (that hasn't been used yet) to store the filtered articles",
    "001": "Please select an input collection to use and enter a research prompt",
    "011": "Please select an input collection to use",
    "101": "Please enter a research prompt",
    "110": "Please enter a collection name (that hasn't been used yet) to store the filtered articles",
}

### Layout and Logic when the user enters the relevant fields ###
if not st.session_state.pdf_filtered:
    #  Input the relevant fields - input collection, prompt, and output collection name
    input_collection_name = st.selectbox(
        'Input Collection', chromaUtils.getListOfCollection(), 
        placeholder="Select the collection you would like to use"
    )
    prompt = st.text_input("Research Prompt", placeholder='Enter your research prompt')
    output_collection_name = st.text_input("Output Collection Name", placeholder='e.g. pfa-and-culture', help="It is recommended to pick a name that is similar to your prompt")

    st.markdown('##')
    col1, col2, col3 , col4, col5, col6, col7 = st.columns(7)

    with col4:
        button_placeholder = col4.empty()
        button_placeholder.button("Submit", key="submit")
        
    if st.session_state['submit']:
        start_time = time.time()
        # Check if all 3 options have been filled in
        err_code = str(int(bool(input_collection_name)))+\
                                    str(int(bool(prompt)))+\
                                                str(int(bool(output_collection_name and output_collection_name not in chromaUtils.getListOfCollection())))
        # If input_collection_name, prompt and output_collection_name has been filled up
        if not err_messages.get(err_code):
            with get_openai_callback() as usage_info:
                try:
                    corrected_input, relevant_output = process_user_input(prompt)
                except:
                    st.error("Processing Error! Please try again!")
                
                total_input_tokens = usage_info.prompt_tokens
                total_output_tokens = usage_info.completion_tokens
                total_cost = usage_info.total_cost
                update_usage_logs(Stage.MISCELLANEOUS.value, corrected_input, total_input_tokens, total_output_tokens, total_cost)
            
                # If the question is deemed as irrelevant
                if (('irrelevant' in relevant_output) or ('relevant' not in relevant_output)):
                    st.error('Please input a relevant prompt')

                # If the output collection name is invalid
                elif not chromaUtils.is_valid_name(output_collection_name):
                    naming_format = """
                    Collection Name format MUST satisfy the following format:\n
                    - The length of the name must be between 3 and 63 characters.\n
                    - The name must start and end with a lowercase letter or a digit, and it can contain dots, dashes, and underscores in between.\n
                    - The name must not contain two consecutive dots.\n
                    - The name must not be a valid IP address."""
                    st.error(naming_format)

                # If no errors in input, continue processing
                else:
                    button_placeholder.empty()

                    # Initialization of progress bar
                    PARTS_ALLOCATED_IND_ANALYSIS = 0.5 
                    PARTS_ALLOCATED_AGG_ANALYSIS = 0.3
                    PARTS_ALLOCATED_COPY = 0.2
                    st.warning("DO NOT navigate to another page while the filtering is in progress!")
                    progressBar1 = st.progress(0, text="Processing documents...")
                    st.markdown(f'<small style="text-align: left; color: Black;">Prompt taken in as:  <em>"{corrected_input}</em>"</small>', unsafe_allow_html=True)
                    time.sleep(2)

                    # Get the findings for each individual article along with the table visual
                    ind_findings, findings_visual = ind_analysis_main(corrected_input, input_collection_name, progressBar1)
                    ind_findings.to_excel("output/pdf_analysis_results.xlsx", index=False)
                    time.sleep(2)

                    # Get the individual articles that are deemed relevant to load it into aggregated analysis
                    rel_ind_findings  = ind_findings[ind_findings["Answer"].str.lower() == "yes"]
                    agg_findings= "No Relevant Articles Found" 
                    if rel_ind_findings.shape[0] > 0:
                        agg_findings = agg_analysis_main(rel_ind_findings, progressBar1)
                    
                        # Create output collection containing articles deemed relevant
                        rel_file_names = rel_ind_findings['Article Name'].values.tolist()
                        executor, futures = copyCollection(input_collection_name, output_collection_name, rel_file_names)
                        numDone, numFutures = 0, len(futures)
                        for future in as_completed(futures):
                            result = future.result()
                            numDone += 1
                            progress = float(numDone/numFutures)*PARTS_ALLOCATED_COPY+(PARTS_ALLOCATED_IND_ANALYSIS+PARTS_ALLOCATED_AGG_ANALYSIS)
                            progressBar1.progress(progress,text="Creating collection...")
                    
                    # Store tables & results for display
                    st.session_state.pdf_filtered = corrected_input
                    st.session_state.pdf_ind_fig1 = findings_visual
                    st.session_state.pdf_ind_fig2 = ind_findings
                    st.session_state.pdf_agg_fig = agg_findings

                    # Track and display time taken for processing
                    end_time = time.time()
                    time_taken_seconds = end_time - start_time
                    time_taken_hours_minute_seconds =  time.strftime("%H:%M:%S", time.gmtime(time_taken_seconds))
                    st.session_state.pdf_filtering_time = time_taken_hours_minute_seconds
                    print(f'Time taken in seconds is {time_taken_seconds} seconds')
                    print(f'Time taken in hours minutes and seconds is {time_taken_hours_minute_seconds}')
                    st.success(f'Successful! Time taken: {time_taken_hours_minute_seconds}')

                    st.experimental_rerun()
        else:
           st.error(err_messages[err_code]) 

 ### After processing is completed ###          
if st.session_state.pdf_filtered:
    # Display prompt for user reference
    st.subheader("Prompt")
    st.markdown(st.session_state.pdf_filtered) 

    # Display time taken
    st.subheader("Time Taken")
    st.markdown(st.session_state.pdf_filtering_time)

    st.subheader("Results")

    # Create & display metric card visualisations
    num_relevant_articles = len(get_yes_pdf_filenames(st.session_state.pdf_ind_fig2))
    num_articles = st.session_state.pdf_ind_fig2.shape[0]

    # tyling for metric cards
    st.markdown("""
    <style>
    div[data-testid="metric-container"] {
    background-color: rgba(28, 131, 225, 0.1);
    border: 1px solid rgba(28, 131, 225, 0.1);
    padding: 5% 5% 5% 10%;
    border-radius: 5px;
    color: rgb(30, 103, 119);
    overflow-wrap: break-word;
    }

    div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
    overflow-wrap: break-word;
    white-space: break-spaces;
    color: red;
    }
                
    div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div p {
    font-size: 150% !important;
    }
    </style>
    """
    , unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col2:
        st.metric("Articles Analysed", num_articles)
    
    with col4:
        st.metric("Relevant Articles", num_relevant_articles)

    st.text("")
    st.text("")
    st.text("")

    # Display Result Table
    with open("output/pdf_analysis_results.xlsx", 'rb') as my_file:
        st.download_button(label = 'Download Excel', data = my_file, file_name='pdf_analysis_results.xlsx', mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    st.plotly_chart(st.session_state.pdf_ind_fig1, use_container_width=True)

    # Display Key Themes
    st.subheader("Key Themes")
    st.markdown(st.session_state.pdf_agg_fig)

    reupload_button = st.button('Ask another question')
    if reupload_button:
        st.session_state.pdf_filtered = False
        st.experimental_rerun()