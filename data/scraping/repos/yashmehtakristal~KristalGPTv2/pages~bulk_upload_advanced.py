# All imports

import streamlit as st
from streamlit_extras.app_logo import add_logo
from st_pages import Page, Section, add_page_title, show_pages, hide_pages

# Setting page config & header
st.set_page_config(page_title = "Kristal Retriever", page_icon = "📖", layout = "wide")
st.header("📖 Kristal Retriever")

import openai
import os
import tempfile
from tempfile import NamedTemporaryFile
from streamlit_extras.app_logo import add_logo
from llama_index.readers.schema.base import Document
from bundle import no_embeddings_process_documents_loop_advanced, embeddings_process_documents_loop_advanced
from core.output import download_data_as_excel_link, download_data_as_csv_link
from core.loading import display_document_from_uploaded_files
from core.chroma import download_embedding_zip, st_server_file, upload_zip_files


add_logo("https://assets-global.website-files.com/614a9edd8139f5def3897a73/61960dbb839ce5fefe853138_Kristal%20Logotype%20Primary.svg")


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY
openai_api_key = OPENAI_API_KEY


# Error handling for OpenAI API key
if not openai_api_key:
    st.warning(
        "There is something wrong with the API Key Configuration."
        "Please check with creator of the program (OpenAI keys can be found at https://platform.openai.com/account/api-keys)"
    )

st.session_state.username = st.session_state.username

def change_states():
    st.session_state.logged_out = True
    st.session_state.logged_in = False
    st.session_state.password_match = None

# Initializing session states
if "load_prompt_result_selector_state" not in st.session_state:
    st.session_state.load_prompt_result_selector_state = False

if "output_response" not in st.session_state:
    st.session_state.output_response = 0

if "llm_prompts_to_use" not in st.session_state:
    st.session_state.llm_prompts_to_use = 0

if "context_with_max_score_list" not in st.session_state:
    st.session_state.context_with_max_score_list = 0

if "file_path_metadata_list" not in st.session_state:
    st.session_state.file_path_metadata_list = 0

if "source_metadata_list" not in st.session_state:
    st.session_state.source_metadata_list = 0

if "prompt_result_selector" not in st.session_state:
    st.session_state.prompt_result_selector = 0

if "process_documents" not in st.session_state:
    st.session_state.process_documents = False


# Display app only if user is logged in
if st.session_state.logged_in is True and st.session_state.logout is False:

    st.sidebar.subheader(f'Welcome {st.session_state.username}')

    #st.session_state.Authenticator.logout('Log Out', 'sidebar')
    # logout_button = st.session_state.Authenticator.logout('Log Out', 'sidebar')
    logout_button = st.sidebar.button("Logout", on_click = change_states)

    # Check embeddings
    check_embeddings = st.radio(label = "Do you have saved embeddings?", options = ["Yes", "No"], index = None, help = "Embeddings are saved files created by ChromaDB", disabled=False, horizontal = False, label_visibility="visible")


    # User does not have embeddings they can use
    if check_embeddings == "No":

        master_folder, chroma_file_path, chroma_file_name = st_server_file()

        # File uploader section for pdfs
        uploaded_files = st.file_uploader(
        "Upload your pdf documents",
        type=["pdf"],
        help="You can upload multiple files."
        "Please note that scanned documents are not supported yet!",
        accept_multiple_files = True
    )


        # File uploader section for xlsx
        uploaded_xlsx_files = st.file_uploader(
        "Upload a xlsx file",
        type=["xlsx"],
        help="Please upload the excel file. Make sure it is in the appropriate format. Check the [name] sidebar for more details about the format",
        accept_multiple_files = False
    )
        
        # Fund name
        fund_variable = st.text_input(
            label = "Fund name:",
            value = None,
            max_chars = None,
            type = "default",
            help = "This will be used to replace the word, fund, in certain prompts",
            placeholder = '''Please input the exact, full fund name. Example: FRANKLIN US GOVERNMENT "A" INC''',
            disabled = False,
            label_visibility = "visible"
        )

        # Model selection
        MODEL_LIST = ["gpt-3.5-turbo", "gpt-4"]

        # Select model to use (type hint as string)
        model: str = st.selectbox(label = "Model", options = MODEL_LIST, index = 1, help = "Please select the appropriate LLM model you want to use. Refer to https://platform.openai.com/docs/models/overview for the model details", placeholder = "Please choose an option ...")

        # Nodes to retrieve slider
        nodes_to_retrieve = st.slider(label = "Please select the number of nodes to retrieve from LLM", min_value = 0, max_value = 5, value = 3, step = 1,
                help =
                '''   
                Nodes to retrieve is simply how many nodes LLM will consider in giving output.
                Higher the number of nodes, greater the accuracy but more costlier it will be, and vice-versa.
                I'd recommend setting an even balance (hence, set a default value of 3)
                ''',
                disabled = False,
                label_visibility = "visible")

        # Temperature slider
        temperature = st.slider(label = "Please select temperature of the LLM", min_value = 0.0, max_value = 1.0, value = 0.2, step = 0.1,
                help =
                '''   
                Temperature is a parameter that controls the “creativity” or randomness of the text generated by GPT-3.
                A higher temperature (e.g., 0.7) results in more diverse and creative output, while a lower temperature (e.g., 0.2) makes the output more deterministic and focused.
                Look at this page for more details: https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api-a-few-tips-and-tricks-on-controlling-the-creativity-deterministic-output-of-prompt-responses/172683
                ''',
                disabled = False,
                label_visibility = "visible")
        
        # Timeout for requests slider
        request_timeout = st.slider(label = "Please select the request timeout (in seconds) of the LLM", min_value = 0, max_value = 600, value = 120, step = 60,
                help =
                '''   
                Request timeout is the timeout for requests to OpenAI completion API
                A higher number means you wait for a longer time before the request times out and vice versa.
                Note, too high a number means you wait too long and too low a number means you don't give it chance to retry.
                I'd recommend striking a balance but leaning a bit more towards lower side (hence, default is 120 seconds)
                ''',
                disabled = False,
                label_visibility = "visible")
        
        # Maximum retries slider
        max_retries = st.slider(label = "Please select the maximum retries of the LLM", min_value = 0, max_value = 15, value = 5, step = 1,
                help =
                '''   
                This is maximum number of retries LLM will make in case it reaches a failure
                A higher number means you allow it for more failure and vice versa.
                Note, too high a number means you wait too long and too low a number means you don't give it chance to retry.
                I'd recommend striking an even balance (hence, default is 5 retries)
                ''',
                disabled = False,
                label_visibility = "visible")
        

        # Sleep function slider
        sleep = st.slider(label = "Please select the amount of time you want LLM to sleep before executing next prompt (in seconds)", min_value = 0, max_value = 60, value = 8, step = 1,
                help =
                '''   
                This is amount of time our LLM will sleep before executing next prompt.
                This is done primarily to avoid ratelimit errors and any failure that might interrupt the code.
                A higher number means you wait for more time and have less chances of hitting ratelimit errors, and vice versa,
                I'd recommend leaning more towards a lower number (hence, default is 8 seconds
                Besides this, there is also another safety check that will conduct exponential waiting between 1 and 20 seconds, for maximum 6 retries (using tenacity library)
                )
                ''',
                disabled = False,
                label_visibility = "visible")
        
        # Advanced options:
        # Return_all_chunks: Shows all chunks retrieved from vector search
        # Show_full_doc: Displays parsed contents of the document
        with st.expander("Advanced Options"):
            return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
            show_full_doc = st.checkbox("Show parsed contents of the document")
            show_tables = st.checkbox("Show tables in dataframe")


        # Error handling for model selection
        if not model:
            st.warning("Please select a model", icon="⚠")
            st.stop()


    # User has embeddings which they can use
    elif check_embeddings == "Yes":

        uploaded_zip_file = upload_zip_files()

        # File uploader section for pdfs
        uploaded_files = st.file_uploader(
        "Upload your pdf documents",
        type=["pdf"],
        help="You can upload multiple files."
        "Please note that scanned documents are not supported yet!",
        accept_multiple_files = True
    )

        # File uploader section for xlsx
        uploaded_xlsx_files = st.file_uploader(
        "Upload a xlsx file",
        type=["xlsx"],
        help="Please upload the excel file. Make sure it is in the appropriate format. Check the [name] sidebar for more details about the format",
        accept_multiple_files = False
    )

        # Fund name
        fund_variable = st.text_input(
            label = "Fund name:",
            value = None,
            max_chars = None,
            type = "default",
            help = "This will be used to replace the word, fund, in certain prompts",
            placeholder = '''Please input the exact, full fund name. Example: FRANKLIN US GOVERNMENT "A" INC''',
            disabled = False,
            label_visibility = "visible"
        )

        # Model selection
        MODEL_LIST = ["gpt-3.5-turbo", "gpt-4"]

        # Select model to use (type hint as string)
        model: str = st.selectbox(label = "Model", options = MODEL_LIST, index = 1, help = "Please select the appropriate LLM model you want to use. Refer to https://platform.openai.com/docs/models/overview for the model details", placeholder = "Please choose an option ...")

        # Temperature slider
        nodes_to_retrieve = st.slider(label = "Please select the number of nodes to retrieve from LLM", min_value = 0, max_value = 5, value = 3, step = 1,
                help =
                '''   
                Nodes to retrieve is simply how many nodes LLM will consider in giving output.
                Higher the number of nodes, greater the accuracy but more costlier it will be, and vice-versa.
                I'd recommend setting an even balance (hence, set a default value of 3)
                ''',
                disabled = False,
                label_visibility = "visible")

        # Temperature slider
        temperature = st.slider(label = "Please select temperature of the LLM", min_value = 0.0, max_value = 1.0, value = 0.2, step = 0.1,
                help =
                '''   
                Temperature is a parameter that controls the “creativity” or randomness of the text generated by GPT-3.
                A higher temperature (e.g., 0.7) results in more diverse and creative output, while a lower temperature (e.g., 0.2) makes the output more deterministic and focused.
                Look at this page for more details: https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api-a-few-tips-and-tricks-on-controlling-the-creativity-deterministic-output-of-prompt-responses/172683
                ''',
                disabled = False,
                label_visibility = "visible")
        
        # Timeout for requests slider
        request_timeout = st.slider(label = "Please select the request timeout (in seconds) of the LLM", min_value = 0, max_value = 600, value = 120, step = 60,
                help =
                '''   
                Request timeout is the timeout for requests to OpenAI completion API
                A higher number means you wait for a longer time before the request times out and vice versa.
                Note, too high a number means you wait too long and too low a number means you don't give it chance to retry.
                I'd recommend striking a balance but leaning a bit more towards lower side (hence, default is 120 seconds)
                ''',
                disabled = False,
                label_visibility = "visible")
        
        # Maximum retries slider
        max_retries = st.slider(label = "Please select the maximum retries of the LLM", min_value = 0, max_value = 15, value = 5, step = 1,
                help =
                '''   
                This is maximum number of retries LLM will make in case it reaches a failure
                A higher number means you allow it for more failure and vice versa.
                Note, too high a number means you wait too long and too low a number means you don't give it chance to retry.
                I'd recommend striking an even balance (hence, default is 5 retries)
                ''',
                disabled = False,
                label_visibility = "visible")
        

        # Sleep function slider
        sleep = st.slider(label = "Please select the amount of time you want LLM to sleep before executing next prompt (in seconds)", min_value = 0, max_value = 60, value = 8, step = 1,
                help =
                '''   
                This is amount of time our LLM will sleep before executing next prompt.
                This is done primarily to avoid ratelimit errors and any failure that might interrupt the code.
                A higher number means you wait for more time and have less chances of hitting ratelimit errors, and vice versa,
                I'd recommend leaning more towards a lower number (hence, default is 8 seconds
                Besides this, there is also another safety check that will conduct exponential waiting between 1 and 20 seconds, for maximum 6 retries (using tenacity library)
                )
                ''',
                disabled = False,
                label_visibility = "visible")
        
        # Advanced options:
        # Return_all_chunks: Shows all chunks retrieved from vector search
        # Show_full_doc: Displays parsed contents of the document
        with st.expander("Advanced Options"):
            return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
            show_full_doc = st.checkbox("Show parsed contents of the document")
            show_tables = st.checkbox("Show tables in dataframe")

        # Error handling for model selection
        if not model:
            st.warning("Please select a model", icon="⚠")
            st.stop()

    # No value inserted for check_embeddings - raise warning
    else:
        st.warning("Please select whether you have embeddings to use or not")
        st.stop()

    # If user clicks on the button process 
    if st.button("Process documents", type = "primary"):

        # User does not have embeddings they can use
        if check_embeddings == "No":
            if uploaded_files and uploaded_xlsx_files:
                    
                # Call bundle function - no_embeddings_process_documents
                output_response, llm_prompts_to_use, context_with_max_score_list, file_path_metadata_list, source_metadata_list, orignal_excel_file, table_dfs, docs = no_embeddings_process_documents_loop_advanced(uploaded_files = uploaded_files, uploaded_xlsx_files = uploaded_xlsx_files, chroma_file_path = chroma_file_path, model = model, nodes_to_retrieve = nodes_to_retrieve, temperature = temperature, request_timeout = request_timeout, max_retries = max_retries, sleep = sleep, return_all_chunks = return_all_chunks, fund_variable = fund_variable)

                with st.expander("Display prompt results & relevant context"):
                    for i in range(len(llm_prompts_to_use)):
                        st.markdown(f"Displaying results for Prompt #{i}: {llm_prompts_to_use[i]}")
                        answer_col, sources_col = st.columns(2)

                        # Displaying answers columns
                        with answer_col:
                            st.markdown("#### Answer")
                            st.markdown(output_response[i])
                        
                        # Displaying sources columns
                        with sources_col:

                            # User selected option to display all chunks from vector search
                            if return_all_chunks is True:

                                # These are lists of corresponding question (as source was list of list)
                                context_to_display = context_with_max_score_list[i]
                                file_path_to_display = file_path_metadata_list[i]
                                source_metadata_to_display = source_metadata_list[i]

                                for chunk in range(nodes_to_retrieve):
                                    st.markdown(context_to_display[chunk])
                                    st.markdown(f"Document: {file_path_to_display[chunk]}")
                                    st.markdown(f"Page Source: {source_metadata_to_display[chunk]}")
                                    st.markdown("---")
                                
                            # User selected option to display only 1 chunk
                            if return_all_chunks is False:
                                
                                # Display particular lists
                                st.markdown(context_with_max_score_list[i])
                                st.markdown(f"Document: {file_path_metadata_list[i]}")
                                st.markdown(f"Page Source: {source_metadata_list[i]}")
                                st.markdown("---")

                # If show full document option is True
                if show_full_doc is True:

                    # Display parsed results in the expander
                    with st.expander("Display parsed documents"):
                        content, content_document_list, content_filename = display_document_from_uploaded_files(uploaded_files)
                        for i in range(len(content_document_list)):

                            st.markdown(f"### File name: {content_filename[i]}")
                            st.markdown(content_document_list[i])

                    
                # If show tables option is True, display it in expander
                if show_tables is True:

                    # Display all parsed tables
                    with st.expander("Display Parsed Tables"):
                        st.markdown(f"Parsed Table results")

                        # st.write(table_dfs)
                        for i in range(len(table_dfs)):
                            st.dataframe(table_dfs[i])


                # Display dataframe and download to excel and csv
                st.markdown("### Bulk Prompt Results")
                st.dataframe(data = orignal_excel_file, use_container_width = True, column_order = None) 
                download_data_as_excel_link(orignal_excel_file = orignal_excel_file)
                download_data_as_csv_link(orignal_excel_file = orignal_excel_file)
                download_embedding_zip(chroma_file_path, zip_filename = "embeddings")

            ## ERROR HANDLING

            elif uploaded_files and not uploaded_xlsx_files:
                st.warning("1) Please upload an excel file", icon="⚠")
                st.stop()


            elif uploaded_xlsx_files and not uploaded_files:
                st.warning("1) Please upload pdf files", icon="⚠")
                st.stop()

            else:
                st.warning(
                    '''
                    1) Please upload the pdf files
                    2) and upload the excel files''',
                    icon="⚠")
                st.stop()


        # User does not have embeddings they can use
        elif check_embeddings == "Yes":
            if uploaded_xlsx_files:
                # Call bundle function - no_embeddings_process_documents
                output_response, llm_prompts_to_use, context_with_max_score_list, file_path_metadata_list, source_metadata_list, orignal_excel_file, table_dfs, docs = embeddings_process_documents_loop_advanced(uploaded_files = uploaded_files, uploaded_xlsx_files = uploaded_xlsx_files, model = model, nodes_to_retrieve = nodes_to_retrieve, temperature = temperature, request_timeout = request_timeout, max_retries = max_retries, sleep = sleep, return_all_chunks = return_all_chunks, fund_variable = fund_variable, uploaded_zip_file = uploaded_zip_file)
                
                with st.expander("Display prompt results & relevant context"):

                    for i in range(len(llm_prompts_to_use)):
                        st.markdown(f"Displaying results for Prompt #{i}: {llm_prompts_to_use[i]}")
                        answer_col, sources_col = st.columns(2)

                        # Displaying answers columns
                        with answer_col:
                            st.markdown("#### Answer")
                            st.markdown(output_response[i])
                        
                        # Displaying sources columns
                        with sources_col:
                            if return_all_chunks is True:

                                context_to_display = context_with_max_score_list[i]
                                file_path_to_display = file_path_metadata_list[i]
                                source_metadata_to_display = source_metadata_list[i]

                                for chunk in range(nodes_to_retrieve):
                                    st.markdown(context_to_display[chunk])
                                    st.markdown(f"Document: {file_path_to_display[chunk]}")
                                    st.markdown(f"Page Source: {source_metadata_to_display[chunk]}")
                                    st.markdown("---")
                                
                            if return_all_chunks is False:
                                st.markdown(context_with_max_score_list[i])
                                st.markdown(f"Document: {file_path_metadata_list[i]}")
                                st.markdown(f"Page Source: {source_metadata_list[i]}")
                                st.markdown("---")

                if show_full_doc is True:

                    with st.expander("Display parsed documents"):
                        content, content_document_list, content_filename = display_document_from_uploaded_files(uploaded_files)
                        for i in range(len(content_document_list)):
                            st.markdown(f"### File name: {content_filename[i]}")
                            st.markdown(content_document_list[i])
                    
                if show_tables is True:

                    # Display all parsed tables
                    with st.expander("Display Parsed Tables"):
                        st.markdown(f"Parsed Table results")
                        for i in range(len(table_dfs)):
                            st.dataframe(table_dfs[i])

                # Display download options
                st.markdown("### Bulk Prompt Results")
                st.dataframe(data = orignal_excel_file, use_container_width = True, column_order = None) 
                download_data_as_excel_link(orignal_excel_file = orignal_excel_file) 
                download_data_as_csv_link(orignal_excel_file = orignal_excel_file)
            else:
                st.warning("1) Please upload the excel files", icon="⚠")
                st.stop()

else:
    st.info("Seems like you are not logged in. Please head over to the Login page to login", icon="ℹ️")
