# All imports

import streamlit as st
from streamlit_extras.app_logo import add_logo
from st_pages import Page, Section, add_page_title, show_pages, hide_pages

# Setting page config & header
st.set_page_config(page_title = "Kristal Retriever", page_icon = "📖", layout = "wide", initial_sidebar_state = "expanded")
st.header("📖 Kristal Retriever")

# Hide particular pages if not logged in
if not st.session_state.logged_in:
    hide_pages(["Bulk Upload - Basic", "Bulk Upload - Advanced", "Q&A - Basic", "Q&A - Advanced"])

# Hide particular pages if logged out
if st.session_state.logged_out:
    hide_pages(["Bulk Upload - Basic", "Bulk Upload - Advanced", "Q&A - Basic", "Q&A - Advanced"])

# Add the logo to the sidebar
add_logo("https://assets-global.website-files.com/614a9edd8139f5def3897a73/61960dbb839ce5fefe853138_Kristal%20Logotype%20Primary.svg")


import openai
import os
import tempfile
from tempfile import NamedTemporaryFile
from database_helper_functions import sign_up, fetch_users
import streamlit_authenticator as stauth

## Importing functions

# from ui import (
#     is_query_valid,
#     display_file_read_error,
# )

# from bundle import no_embeddings_process_documents, embeddings_process_documents
# from core.loading import read_documents_from_directory, iterate_files_from_directory, save_uploaded_file, read_documents_from_uploaded_files, get_tables_from_uploaded_file, iterate_files_from_uploaded_files, iterate_excel_files_from_directory, iterate_uploaded_excel_files, print_file_details, show_dataframes, iterate_uploaded_excel_file
# from core.pickle import save_to_pickle, load_from_pickle
# from core.indexing import query_engine_function, build_vector_index
# from core.LLM_preprocessing import conditions_excel, extract_fund_variable, prompts_to_substitute_variable, storing_input_prompt_in_list
# from core.querying import recursive_retriever_old, recursive_retriever
# from core.LLM_prompting import individual_prompt, prompt_loop
# from core.PostLLM_prompting import create_output_result_column, create_output_context_column, intermediate_output_to_excel
# from core.parsing import create_schema_from_excel, parse_value
# from core.Postparsing import create_filtered_excel_file, final_result_orignal_excel_file, reordering_columns
# from core.Last_fixing_fields import find_result_fund_name, find_result_fund_house, find_result_fund_class, find_result_currency, find_result_acc_or_inc, create_new_kristal_alias, update_kristal_alias, update_sponsored_by, update_required_broker, update_transactional_fund, update_disclaimer, update_risk_disclaimer, find_nav_value, update_nav_value 
# from core.output import output_to_excel, download_data_as_excel, download_data_as_csv

# def login_callback():
#     st.session_state.logged_out = True
#     st.session_state.logged_in = False

# st.write(st.session_state.logged_out, st.session_state.logged_in)

# let User see app if logged in = True & logged out = False
if st.session_state.logged_in is True and st.session_state.logout is False:

    st.sidebar.subheader(f'Welcome {st.session_state.username}')

    #st.session_state.Authenticator.logout('Log Out', 'sidebar')
    logout_button = st.session_state.Authenticator.logout('Log Out', 'sidebar')

    # If user has clicked logged_out button, update the state variables
    if logout_button:
        st.session_state.logged_out = True
        st.session_state.logged_in = False

        # st.write("Before Rerun")
        # st.write(st.session_state.logged_out, st.session_state.logged_in)
        # st.write("XXXX")

        st.rerun()


    # Display Markdown of the main page
    st.markdown(
    '''
    This section will give more information about Kristal GPT. 

    This application has 2 main features (Bulk Upload and Q&A). Moreover, it has two high-level categorization (Basic, Advanced) 

    Here is the simple categorization of the aforementioned:

    - Basic
        - Bulk Upload - Basic
        - Q&A - Basic
    - Advanced
        - Bulk Upload - Advanced
        - Q&A - Advanced

    ### Features explanation

    ***Bulk Upload:***

    This feature allows the user to upload an excel file (or select a template) containing the list of prompts, along with other relevant fields.

    ***Q&A:***

    This feature allows the user to input prompts individually, as if they are "chatting" with the uploaded documents.

    ### Categorization

    ***Basic:***

    The Basic version of the application has the minimum features required to successfully run the application. These are:

    1. Option to save embeddings for current iteration/load saved embeddings
    2. Specifying the folder for the embeddings
    3. Uploading the pdf files, as well as the excel files.
    4. Displaying the results as a dataframe
    5. Providing option to download displayed dataframe as a CSV file or Excel file

    ***Advanced:***

    The Advanced version of the application has the same features as the basic, with the addition of the following:

    1. Select which LLM model to use
    2. Select the number of nodes to retrieve from LLM (during vector search)
    3. Select the temperature parameter of LLM
    4. Select the request timeout (in seconds) of LLM
    5. Select the maximum retries of LLM
    6. Select the amount of time for LLM to wait before executing next prompt (in loop)
    7. Select whether to display all chunks retrieved from vector search (If no, i.e. default value, will display the chunk that has highest score)
    8. Select to show the parsed contents of the document
    9. Select to show all tables parsed from the pdf document
    '''
    )

else:
    st.info("Seems like you are not logged in. Please head over to the Login page to login", icon="ℹ️")
