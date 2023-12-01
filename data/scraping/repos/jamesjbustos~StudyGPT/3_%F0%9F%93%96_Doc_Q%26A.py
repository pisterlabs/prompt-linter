# Import necessary libraries
import streamlit as st
import os
import pinecone
from llama_index import (download_loader, LLMPredictor,
                         PromptHelper, ServiceContext, GPTPineconeIndex)
from langchain import OpenAI

# Streamlit page configurations and title
st.set_page_config(
    page_title="StudyGPT",
    page_icon=":mortar_board:",
    initial_sidebar_state = "collapsed"
)
st.title("📖 Doc Q&A")
st.caption("✨ Your personal document assistant - upload and start asking questions!")

index = None

# Load API Key
api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_enviroment = st.secrets["PINECONE_ENVIRONMENT"]

# ------ Initialize Session State ------
if 'doc_response' not in st.session_state:
    st.session_state.doc_response = ''
    
if 'placeholder_initialized' not in st.session_state:
    st.session_state.placeholder_initialized = False

# ------ Helper functions ------
def save_uploaded_file(uploadedfile):
    with open(os.path.join("data", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())

def remove_all_files_in_directory(directory):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

# ------ Load and index document ------
uploaded_file = st.file_uploader(" ", accept_multiple_files=False,
                                  label_visibility='collapsed', type=['pdf', 'docx', 'txt'])

if uploaded_file is not None:
    # Create data folder if it doesn't exist
    if not os.path.exists('./data'):
        os.mkdir('./data')
    else:
        # Remove all files in the data directory
        remove_all_files_in_directory('./data')

    # Save uploaded file to doc_path
    save_uploaded_file(uploaded_file)

    # Load and index document
    SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
    loader = SimpleDirectoryReader('data', recursive=True, exclude_hidden=True)
    documents = loader.load_data()

    # Pinecone intialization
    pinecone.init(
        api_key=pinecone_api_key, 
        environment=pinecone_enviroment
    )
    pinecone_index = pinecone.Index("studygpt-index")

    # Define llm and index
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))
    prompt_helper = PromptHelper(max_input_size=4096, num_output=256, max_chunk_overlap=20)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    metadata_filters = {"title": uploaded_file.name}
    # Create the GPTPineconeIndex
    index = GPTPineconeIndex.from_documents(
        documents,
        pinecone_index=pinecone_index,
        metadata_filters=metadata_filters,
        service_context=service_context,
        add_sparse_vector=True,
    )


if index is not None:
    
    if not st.session_state.placeholder_initialized:
        # Add a placeholder for the output
        output_placeholder = st.markdown("🤖 **AI:** Have questions about this document? I'm here to help! Just ask me your questions, and I'll find the relevant information within the text.\n\n")
        st.session_state.placeholder_initialized = True
    else:
        output_placeholder = st.empty()
    
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([10, 1])
        user_prompt = col1.text_area(" ", max_chars=2000, key="prompt",
                                      placeholder="Type your question here...", label_visibility="collapsed")
        submitted = col2.form_submit_button("💬")

    if submitted and user_prompt:
        with st.spinner("💭 Waiting for response..."):
            st.session_state.doc_response = str(index.query(user_prompt)).strip()
        response_md = f"🤓 **YOU:** {user_prompt}\n\n🤖 **AI:** {st.session_state.doc_response}\n\n"
        output_placeholder.markdown(response_md)
