import streamlit as st
from streamlit_extras.app_logo import add_logo
import pandas as pd
from langchain.callbacks import get_openai_callback
import time

# Build path from working directory and add to system paths to facilitate local module import
import os, sys
sys.path.append(os.path.join(os.getcwd(), "ChromaDB"))
sys.path.append(os.path.join(os.getcwd(), "analysis"))
sys.path.append(os.path.join(os.getcwd(), "cost_breakdown"))
sys.path.append(os.path.join(os.getcwd(), "Miscellaneous"))

from chromaUtils import getCollection, getDistinctFileNameList, getListOfCollection
from Freeform_Analysis import get_llm_response, parse_source_docs, get_pdf_analysis_table
from update_cost import update_usage_logs, Stage
from User_Input_Cleaning import run_spell_check

st.set_page_config(layout="wide")
add_logo("images/temp_logo.png", height=100)


st.markdown("<h1 style='text-align: left; color: Black;'>PDF Analysis</h1>", unsafe_allow_html=True)
st.markdown('#')

if 'pdf_analysis_prompt' not in st.session_state:
    st.session_state.pdf_analysis_prompt = False
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'pdf_analysis_time' not in st.session_state:
    st.session_state.pdf_analysis_time = None


if not st.session_state.pdf_analysis_prompt:
    # Select collection to analyse
    input_collection_name = st.selectbox(
        'Input Collection', getListOfCollection(), 
        placeholder="Select the collection you would like to use"
    )
    # Provide option to analyse single article
    if input_collection_name:
        selected_article_name = False
        analyse_single_article = st.toggle("Analyse single article", value=False, help="Select only 1 article from input collection to analyse. All articles in the collection will be analysed by default.")
        if analyse_single_article:
            selected_article_name = st.selectbox(
                'Select Article', getDistinctFileNameList(input_collection_name), 
                placeholder="Select the article you would like to analyse"
            )

    # Get user prompt
    input = st.text_input("Research Prompt", placeholder='Enter your research prompt')
         
    # Configure pipeline with additional instructions
    num_chunks_retrieved = 3
    additional_prompt_inst = ""
    provide_additional_inst = st.toggle("Provide additional instructions", value=False, help="Default settings will be used if no additional configurations made")
    if provide_additional_inst:
        col1, col2 = st.columns(2)
        with col1:
            num_chunks_retrieved = st.slider("Number of article chunks to feed to LLM", min_value=1, max_value=10, value=3, 
                                                help="Select number of relevant chunks from the article for the LLM to analyse. More chunks fed will incur higher cost and processing time.")
        additional_prompt_inst = st.text_input("Additional Prompt Instructions", placeholder='Enter your instructions (e.g. limit output to 3 sentences). Leave blank if there is none to add.',
                                               help="Additional prompt instructions will be appended to the prompt sent to the LLM")

    # Button to start analysis
    st.markdown('##')
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col4:
        button_placeholder = col4.empty()
        button_placeholder.button("Submit", key="submit_pdf_analysis_prompt")

    # Run if "Submit" button is clicked
    if st.session_state['submit_pdf_analysis_prompt']:
        start_time = time.time()
        # Run if user has selected a collection of PDFs to analyse
        if input_collection_name:
            # Run if user has included a prompt
            if input:
                if selected_article_name:
                    # Use the single article selected
                    article_title_list = [selected_article_name]
                else:
                    # Get list of article titles in the collection
                    article_title_list = getDistinctFileNameList(input_collection_name)
                total_num_articles = len(article_title_list)
            
                # Run if number of articles > 0
                if total_num_articles > 0:
                    st.warning("DO NOT navigate to another page while the analysis is in progress!")
                    progressBar = st.progress(0, text="Analysing...")

                    # Connect to selected database collection
                    db = getCollection(input_collection_name)

                    # Initialise holder lists to temporarily store output
                    response_list = ['']*total_num_articles
                    source_docs_content_list = ['']*total_num_articles
                    source_docs_page_num_list = ['']*total_num_articles
                    # Holder list to store article titles with error in obtaining output
                    article_error_list = []

                    # Check and correct wrong spelling
                    with get_openai_callback() as usage_info:
                        try: 
                            input = run_spell_check(input)
                            st.markdown(f'<small style="text-align: left; color: Black;">Prompt taken in as:  <em>"{input}</em>"</small>', unsafe_allow_html=True)
                        except:
                            progressBar.empty()
                            st.error("Processing Error! Please try again!")
                        update_usage_logs(Stage.MISCELLANEOUS.value, input, usage_info.prompt_tokens, usage_info.completion_tokens, usage_info.total_cost)
                    
                    # Run PDF Analysis
                    button_placeholder.empty() # remove Submit button when analysis is in progress
                    with get_openai_callback() as usage_info:
                        for i in range(total_num_articles):
                            try: 
                                article_title = article_title_list[i]
                                # Make LLM call to get response
                                response, source_docs = get_llm_response(db, input, article_title, num_chunks_retrieved, additional_prompt_inst)
                                # Record response and source documents
                                response_list[i] = response
                                source_docs_content_list[i], source_docs_page_num_list[i] = parse_source_docs(source_docs)
                            except Exception:
                                # Track articles that had error in obtaining response
                                article_error_list.append(article_title)
                            # Update progress
                            progress_display_text = f"Analysing: {i+1}/{total_num_articles} articles completed"
                            progressBar.progress((i+1)/total_num_articles, text=progress_display_text)
                                
                        pdf_analysis_output_df = pd.DataFrame({"article": article_title_list, "answer": response_list, "page_ref": source_docs_page_num_list,
                                                               "source_docs_contents": source_docs_content_list})
                        # Store dataframe as Excel file in local output folder
                        pdf_analysis_output_df.to_excel("output/pdf_analysis_results.xlsx", index=False)
                            
                        # Display success message
                        progressBar.empty()
                        st.success(f"Analysis Complete")
                        # Display error message if there are articles that cannot be analysed due to error
                        if len(article_error_list) > 0:
                            st.error("Error in extracting output for the articles below")
                            with st.expander("Articles with error:"):
                                for article_title in article_error_list:
                                    st.markdown(f"- {article_title}")
                        
                        # Display time taken
                        end_time = time.time()
                        time_taken_seconds = end_time - start_time
                        time_taken_hours_minute_seconds =  time.strftime("%H:%M:%S", time.gmtime(time_taken_seconds))
                        st.session_state.pdf_analysis_time = time_taken_hours_minute_seconds
                        print(f'Time taken in seconds is {time_taken_seconds} seconds')
                        print(f'Time taken in hours minutes and seconds is {time_taken_hours_minute_seconds}')
                        
                        # Update usage info
                        update_usage_logs(Stage.PDF_ANALYSIS.value, input, 
                            usage_info.prompt_tokens, usage_info.completion_tokens, usage_info.total_cost)

                        st.session_state.pdf_analysis_prompt = input
                        st.session_state.pdf_analysis_collection = input_collection_name
                        st.experimental_rerun()
                else:
                    st.error("You have no PDF articles in this collection")
            else:
                st.error("Please input a prompt")
        else: 
            st.error("Please select a collection. If a collection has not been created, please use the My Collections page to do so.")
       
else:
    st.subheader("Prompt")
    st.markdown(st.session_state.pdf_analysis_prompt)

    st.subheader("Time Taken")
    st.markdown(st.session_state.pdf_analysis_time)

    st.subheader("Collection")
    st.markdown(st.session_state.pdf_analysis_collection)

    # Read output Excel file
    pdf_analysis_df = pd.read_excel("output/pdf_analysis_results.xlsx")
    st.subheader("Results")

    # Download output as Excel file
    with open("output/pdf_analysis_results.xlsx", 'rb') as my_file:
        st.download_button(label="Download Excel",
                            # Store output results in a csv file
                            data=my_file,
                            # Query appended at end of output file name
                            file_name=f'analysis_output [{st.session_state.pdf_analysis_collection}] [{st.session_state.pdf_analysis_prompt}].xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    
    # Display output
    fig1 = get_pdf_analysis_table(pdf_analysis_df)
    pdf_analysis_table_height = min(pdf_analysis_df.shape[0]*250, 800)
    fig1.update_layout(margin_autoexpand=True, height=pdf_analysis_table_height)
    st.session_state.pdf_analysis_table = fig1
    st.plotly_chart(st.session_state.pdf_analysis_table, use_container_width=True)
    
    # Repeat process with another question
    retry_button = st.button('Ask another question')
    if retry_button:
        st.session_state.pdf_analysis_prompt = False
        st.experimental_rerun()