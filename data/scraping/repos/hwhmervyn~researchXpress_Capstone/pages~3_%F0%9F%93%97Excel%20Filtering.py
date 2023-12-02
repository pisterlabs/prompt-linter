import streamlit as st
from streamlit_extras.app_logo import add_logo
from concurrent.futures import as_completed
from langchain.callbacks import get_openai_callback
import pandas as pd
import time


import sys,os

workingDirectory = os.getcwd()
chromaDirectory = os.path.join(os.getcwd(), "ChromaDB")
costDirectory = os.path.join(os.getcwd(), "cost_breakdown")
cleanDirectory = os.path.join(os.getcwd(), "Miscellaneous")

if chromaDirectory not in sys.path:
    sys.path.append(chromaDirectory)
if costDirectory not in sys.path:
    sys.path.append(costDirectory)
if cleanDirectory not in sys.path:
    sys.path.append(cleanDirectory)

from filterExcel import filterExcel, getOutputDF, generate_visualisation, clean_findings_df, getYesExcel
from User_Input_Cleaning import process_user_input
from update_cost import update_usage_logs, Stage

st.set_page_config(layout="wide")
add_logo("images/temp_logo.png", height=100)

st.markdown("<h1 style='text-align: left; color: Black;'>Excel Filtering</h1>", unsafe_allow_html=True)
st.markdown('#')


if 'filtered' not in st.session_state:
    st.session_state.filtered = False
if 'excel_filtering_time' not in st.session_state:
    st.session_state.excel_filtering_time = None


if not st.session_state.filtered:
    input = st.text_input("Research Prompt", placeholder='Enter your research prompt')

    st.markdown('##')
    uploaded_file = st.file_uploader("Upload your Excel file here", type=['xlsx'], help='Upload an excel file that contains a list of research article titles and their abstracts')

    st.markdown('##')
    col1, col2, col3 , col4, col5, col6, col7 = st.columns(7)

    with col4:
        button_placeholder = col4.empty()
        button_placeholder.button("Submit", key="submit_excel")

    if st.session_state['submit_excel']:
        start_time = time.time()
        # If Research Prompt is Filled and Excel File is not uploaded 
        if input and not uploaded_file: 
            st.error("Please upload an excel file")
        # If Research Prompt is not Filled and Excel File is uploaded 
        elif not input and uploaded_file:
            st.error("Please enter a research prompt")
        # If Research Prompt is not Filled and Excel File is not uploaded 
        elif not input or not uploaded_file:
            st.error("Please enter a research prompt and upload an excel file")
        # Research Prompt is Filled and Excel File is uploaded
        else:
            # Check if Uploaded Excel Format is valid
            excel_format = True
            try:
                excel_format_checker = pd.read_excel(uploaded_file).dropna(how='all')[['DOI','TITLE','ABSTRACT']]
            except KeyError:
                excel_format = False
            # Excel Format is valid
            if excel_format: 
                # Research prompt cleaning and relevancy checker
                with get_openai_callback() as usage_info:
                    try:
                        corrected_input, relevant_output = process_user_input(input)
                    except:
                        st.error("Processing Error! Please try again!")
                    total_input_tokens = usage_info.prompt_tokens
                    total_output_tokens = usage_info.completion_tokens
                    total_cost = usage_info.total_cost
                    update_usage_logs(Stage.MISCELLANEOUS.value, input, total_input_tokens, total_output_tokens, total_cost)
                # If the question is deemed as irrelevant
                if (('irrelevant' in relevant_output) or ('relevant' not in relevant_output)):
                    st.error('Irrelevant Output! Please input a relevant prompt')
                # Question is relevant
                else:
                    st.warning("DO NOT navigate to another page while the filtering is in progress!")
                    button_placeholder.empty()
                    _, futures = filterExcel(uploaded_file,  corrected_input)
                    issues, results, numDone, numFutures, total_input_tokens, total_output_tokens, total_cost = [],[], 0, len(futures), 0, 0, 0
                    progessBar = st.progress(0, text="Article filtering in progress...")
                    st.markdown(f'<small style="text-align: left; color: Black;">Prompt taken in as:  <em>"{corrected_input}</em>"</small>', unsafe_allow_html=True)
                    # OpenAI analyzing relevancy row by row
                    for future in as_completed(futures):
                        row = future.result()
                        total_input_tokens += row[5]
                        total_output_tokens += row[6]
                        total_cost += row[7]
                        results.append(row[0:5])
                        numDone += 1
                        progessBar.progress(numDone/numFutures, text="Article filtering in progress...")
                    end_time = time.time()
                    time_taken_seconds = end_time - start_time
                    time_taken_hours_minute_seconds =  time.strftime("%H:%M:%S", time.gmtime(time_taken_seconds))
                    st.session_state.excel_filtering_time = time_taken_hours_minute_seconds
                    print(f'Time taken in seconds is {time_taken_seconds} seconds')
                    print(f'Time taken in hours minutes and seconds is {time_taken_hours_minute_seconds}')
                    st.success(f'Successful! Time taken: {time_taken_hours_minute_seconds}')
                    update_usage_logs(Stage.EXCEL_FILTERING.value, corrected_input, total_input_tokens,total_output_tokens,total_cost)
                    dfOut = pd.DataFrame(results, columns = ["DOI","TITLE","ABSTRACT","LLM OUTPUT", "jsonOutput"])
                    dfOut = getOutputDF(dfOut)
                    dfOut.to_excel("output/excel_result.xlsx", index=False)
                    st.session_state.filtered = input
                    st.experimental_rerun()
            # Excel Format is not valid
            else:
                st.error('Wrong Excel Format: "TITLE", "ABSTRACT", "DOI" Columns Required')
      

else:
    st.subheader("Prompt")
    st.markdown(st.session_state.filtered)

    st.subheader("Time Taken")
    st.markdown(st.session_state.excel_filtering_time)

    st.subheader("Results")
    
    df = pd.read_excel("output/excel_result.xlsx")
    
    # Summary visualisations (in the form of cards)
    num_relevant_articles = len(getYesExcel(df))
    num_articles = df.shape[0]
    
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

    # Display output
    clean_df = clean_findings_df(df)
    fig = generate_visualisation(clean_df)
    st.session_state.excel_filter_fig1 = fig

    # Download Button
    with open("output/excel_result.xlsx", 'rb') as my_file:
        st.download_button(label = 'Download Excel', data = my_file, file_name='results.xlsx', mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet') 
        
    st.plotly_chart(st.session_state.excel_filter_fig1, use_container_width=True)

    reupload_button = st.button('Reupload another prompt and excel file')
    if reupload_button:
        st.session_state.filtered = False
        st.experimental_rerun()