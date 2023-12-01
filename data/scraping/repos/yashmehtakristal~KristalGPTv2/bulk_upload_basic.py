# All imports

import streamlit as st
from streamlit_extras.app_logo import add_logo
from st_pages import Page, Section, add_page_title, show_pages, hide_pages

# Setting page config & header
st.set_page_config(page_title = "Kristal Retriever", page_icon = "📖", layout = "wide")
st.header("📖 Kristal Retriever")

add_logo("https://assets-global.website-files.com/614a9edd8139f5def3897a73/61960dbb839ce5fefe853138_Kristal%20Logotype%20Primary.svg")

import openai

## Importing functions
from bundle import no_embeddings_process_documents_loop, embeddings_process_documents_loop
from core.chroma import st_server_file, upload_zip_files

### CODE

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

# Display app only if user is logged in
if st.session_state.logged_in is True and st.session_state.logout is False:

    st.sidebar.subheader(f'Welcome {st.session_state.username}')

    #st.session_state.Authenticator.logout('Log Out', 'sidebar')
    # logout_button = st.session_state.Authenticator.logout('Log Out', 'sidebar')
    logout_button = st.sidebar.button("Logout", on_click = change_states)

    check_embeddings = st.radio(label = "Do you have saved embeddings?", options = ["Yes", "No"], index = None, help = "Embeddings are saved files created by ChromaDB", disabled=False, horizontal = False, label_visibility="visible")

    # User does not have embeddings they can use
    if check_embeddings == "No":

        # Obtain chrome_file_path and chroma_file_name
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
        
        # Fund name variable
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
        accept_multiple_files = False)

        # Fund name variable
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
                no_embeddings_process_documents_loop(uploaded_files = uploaded_files, uploaded_xlsx_files = uploaded_xlsx_files, chroma_file_path = chroma_file_path, fund_variable = fund_variable)

            ## Error handling

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

            # Checking if all three conditions are satisfied
            if uploaded_files and uploaded_xlsx_files:

                # Call bundle function - no_embeddings_process_documents
                embeddings_process_documents_loop(uploaded_files = uploaded_files, uploaded_xlsx_files = uploaded_xlsx_files, fund_variable = fund_variable, uploaded_zip_file = uploaded_zip_file)

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


else:
    st.info("Seems like you are not logged in. Please head over to the Login page to login", icon="ℹ️")
