import streamlit as st
import os
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
import tempfile

st.set_page_config(layout='wide', page_title="EduAI Notes", page_icon=":sleep:")

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Input for API key
if 'api_key' not in st.session_state:
    st.session_state['api_key'] = ''
api_key = st.sidebar.text_input('Enter your OpenAI API Key:', value=st.session_state['api_key'], type='password')
# Save the API key to the session state
st.session_state['api_key'] = api_key
# Check if the API key is empty and display a warning if it is
if api_key == '':
    st.warning('You have not entered your API key!')
else:
    os.environ['OPENAI_API_KEY'] = api_key
    llm = OpenAI(temperature=0.1, verbose=True)
    embeddings = OpenAIEmbeddings()

    st.title('📓 📝 Notes Tool')
    st.write("Efficiently learn from your notes by submitting a file in pdf format: ")

    # File uploader to accept PDF files
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(uploaded_file.read())
            tmp_path = tmpfile.name

        # Create and load PDF Loader with the temporary file path
        loader = PyPDFLoader(tmp_path)
        # Split pages from pdf
        pages = loader.load_and_split()
        # Load documents into vector database aka ChromaDB
        store = Chroma.from_documents(pages, embeddings, collection_name='thirdpoint')

        vectorstore_info = VectorStoreInfo(
            name="annual_report",
            description="a banking annual report as a pdf",
            vectorstore=store
        )
        toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

        agent_executor = create_vectorstore_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True
        )

        # Create a text input box for the user
        prompt = st.text_input('Input prompt')

        # If the user hits enter
        if prompt:
            # Then pass the prompt to the LLM
            response = agent_executor.run(prompt)
            # ...and write it out to the screen
            st.write(response)

            # With a streamlit expander
            with st.expander('Search Document'):
                # Find the relevant pages
                search = store.similarity_search_with_score(prompt)
                # Write out the first
                st.write(search[0][0].page_content)
        os.unlink(tmp_path)  # Delete the temporary file
