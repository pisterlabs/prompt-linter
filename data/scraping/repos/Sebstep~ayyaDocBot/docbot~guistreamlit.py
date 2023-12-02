import os
import openai
from dotenv import load_dotenv
import streamlit as st
from llama_index import (
    StorageContext,
    ServiceContext,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.llms import OpenAI
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
from storageLogistics import build_new_storage
from llama_index.vector_stores import SimpleVectorStore
import logging
from localhelpers import (
    parse_response,
    save_response_to_json,
    display_response,
    display_sources,
)
from langchain.chat_models import ChatOpenAI


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("docbot.log"), logging.StreamHandler()],
)

# Set constants
STORAGE_FOLDER = "storage"
OUTPUT_FOLDER = "output"
STORAGE_TYPE = "simple"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Define function to initialize the index
@st.cache_resource
def get_index():
    storage_context = StorageContext.from_defaults(
        persist_dir=f"{STORAGE_FOLDER}/{STORAGE_TYPE}"
    )
    index = load_index_from_storage(storage_context=storage_context)
    return index


# Define function to check whether the API key is valid
def is_api_key_valid():
    try:
        response = openai.Completion.create(
            engine="davinci", prompt="This is a test.", max_tokens=5
        )
    except:
        return False
    else:
        return True


if "openai" not in st.session_state:
    st.session_state["openai"] = None
if st.session_state["openai"] is None:
    if os.getenv("OPENAI_API_KEY"):
        st.session_state["openai"] = os.getenv("OPENAI_API_KEY")
        st.info("API Key read from environment.")
    elif openai.api_key:
        st.session_state["openai"] = openai.api_key

##########################################
# SIDEBAR
##########################################
with st.sidebar:
    st.sidebar.title("Navigation")
    selected_option = st.sidebar.radio("Pages:", ["Chat", "Manage"])

    st.sidebar.divider()

    st.sidebar.header("Setup OpenAI API Key")

    openai_keystring = st.text_input(
        "OpenAI API Key", key="chatbot_api_key", value=st.session_state["openai"]
    )

    updateAPI, removeAPI = st.columns(2)
    with updateAPI:
        if st.button("Update API Key"):
            openai.api_key = openai_keystring
            if is_api_key_valid():
                st.session_state["openai"] = openai_keystring
                st.success("API Key is valid.")
            else:
                st.error("Invalid API Key.")
                openai.api_key = st.session_state["openai"]

    with removeAPI:
        if st.button("Remove API Key"):
            openai.api_key = None
            st.session_state["openai"] = None
            st.experimental_rerun()


##########################################
# MANAGE PAGE
##########################################
if selected_option == "Manage":
    st.title("Manage Index")
    # st.header("Build New Index")

    # st.write("Warning, building a new index takes very long.")

    # if st.button("Build New Index"):
    #     with st.spinner("Building new index..."):
    #         build_new_storage()
    #     st.success("New index built!")

    # st.write(get_index().ref_doc_info)
    st.header("Add File to Index")
    st.write("To upload a new file into the index, use the file uploader below.")
    st.warning("This feature is not yet implemented.", icon="⚠️")
    uploaded_file = st.file_uploader("Choose a file")
    # if uploaded_file is not None:
    #     with open(os.path.join("documents/uploads",uploaded_file.name),"wb") as f:
    #         f.write(uploaded_file.getbuffer())


##########################################
# CHAT PAGE
##########################################
if selected_option == "Chat":
    st.title("Chat Interface")
    if st.session_state["openai"] is None:
        st.warning(
            'Please enter your OpenAI API Key in the sidebar and press "Update API Key".',
            icon="⚠️",
        )
    else:
        st.subheader("LLM Settings")
        col_left, col_right = st.columns(2, gap="medium")

        with col_left:
            model = st.selectbox(
                "Model", ["gpt-3.5-turbo", "gpt-4"], help="Which model to use"
            )

            max_tokens = st.slider(
                label="Max. Tokens",
                min_value=128,
                max_value=2048,
                value=512,
                step=128,
                help="How many tokens to generate",
            )

        with col_right:
            top_k_nodes = st.number_input(
                label="Similarity Top K",
                min_value=1,
                max_value=20,
                value=6,  # set to 6-8 or more for production
                step=1,
                help="How many similar nodes to return and summarize",
            )
            temperature = st.slider(
                label="LLM temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="How creative the LLM should be",
            )

        st.subheader("Chat Interface")
        user_input = st.text_input("Enter your message:", key="prompt")

        if st.button("Send"):
            index = get_index()
            llm = OpenAI(model=model, temperature=temperature, max_tokens=max_tokens)

            service_context = ServiceContext.from_defaults(llm=llm)

            retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k_nodes)

            response_synthesizer = get_response_synthesizer(response_mode="refine")

            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
            )

            with st.chat_message("User", avatar="🙋‍♂️"):
                st.write(user_input)
            with st.spinner("Getting response..."):
                response = query_engine.query(user_input)
            st.success("Response received!")
            parsed_response_dict = parse_response(user_input, response)
            output_json_file = save_response_to_json(
                parsed_response_dict, OUTPUT_FOLDER
            )
            st.toast(f"Saved to:  {output_json_file}", icon="💾")
            with st.chat_message("Bot", avatar="🤖"):
                display_response(parsed_response_dict)
                st.subheader("Sources")
                display_sources(parsed_response_dict)
                st.subheader("Full JSON Object")
                st.json(parsed_response_dict, expanded=True)


# Run Streamlit app
if __name__ == "__main__":
    st.sidebar.text("Streamlit App")
