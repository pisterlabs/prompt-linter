import streamlit as st

st.markdown("Installation instructions 👉 [Github Link](https://github.com/JustinChavez/streamlit-chat-tutorials/tree/main)")
st.markdown("Starting point 👉 [Github Link](https://github.com/JustinChavez/streamlit-chat-tutorials/blob/main/01_streamlit_chatgpt/app.py)")

st.code("""
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os
""", language="python")

st.markdown("### Loading PDF")

st.code("""
#openai.api_key = config("OPENAI_API_KEY")

INDEX = config("INDEX")
""", language="python")

st.markdown("[LangChain PyPDF Loader](https://python.langchain.com/docs/modules/data_connection/document_loaders/how_to/pdf)")
st.code("""
#     return response

file_path = st.file_uploader(label="Upload a PDF file that your chatbot will use", type=['pdf'])

if st.button("Index PDF", disabled=not bool(file_path)):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp_file:
        tmp_file.write(file_path.read())
        # LangChain PyPDF loader
        loader = PyPDFLoader(tmp_file.name)
        pages = loader.load_and_split()

# Define Streamlit Containers
""", language="python")

st.markdown("### Text Splitting")

st.code("""
#pages = loader.load_and_split()

# Split the pages into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
page_chunks = text_splitter.split_documents(pages)
""", language="python")

st.image("sliding_window_example.png")

st.markdown("### Embed PDF with OpenAIEmbeddings + LangChain")
st.markdown("https://openai.com/blog/introducing-text-and-code-embeddings")
st.markdown("https://platform.openai.com/docs/guides/embeddings/what-are-embeddings")
st.markdown("https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/faiss")

st.code("""
#with tempfile.NamedTemporaryFile
    #...
# Embed into FAISS
vectorstore = FAISS.from_documents(page_chunks, OpenAIEmbeddings(openai_api_key=config("OPENAI_API_KEY")))

pdf_index = os.path.splitext(os.path.basename(file_path.name))[0]
local_path = os.path.join(INDEX, pdf_index)

vectorstore.save_local(local_path)

# Save the file name to session state as pdf_index
st.session_state['pdf_index'] = pdf_index
""", language="python")

st.write("Test out uploading and embedding a PDF")
st.markdown("https://bitcoin.org/bitcoin.pdf")

st.code("""
#st.session_state.setdefault('user_message', [])
st.session_state['pdf_index'] = 'bitcoin'
""", language="python")

st.markdown("### Update ChatGPT Completions API to use the PDF context")

st.code("""
# Function for interacting with ChatGPT API
#def generate_response(prompt):
    vectorstore = FAISS.load_local(os.path.join(INDEX, st.session_state['pdf_index']), OpenAIEmbeddings(openai_api_key=config("OPENAI_API_KEY")))
    get_relevant_sources = vectorstore.similarity_search(prompt, k=2)

    template = f"\\n\\nUse the information below to help answer the user's question.\\n\\n{get_relevant_sources[0].page_content}\\n\\n{get_relevant_sources[1].page_content}"
    
    with st.expander("Source 1", expanded=False):
        st.write(get_relevant_sources[0].page_content)
    with st.expander("Source 2", expanded=False):
        st.write(get_relevant_sources[1].page_content)
    system_source_help = {"role": "system", "content": template}

    st.session_state['messages'].append({"role": "user", "content": prompt})

    # Get Previous messages and append context
    to_send = st.session_state['messages'].copy()
    to_send.insert(-1, system_source_help)
""", language="python")

st.code("""

#completion = openai.ChatCompletion.create(
#    model="gpt-3.5-turbo",
    messages=to_send,
#)
""", language="python")

st.write("Test out the chatbot!")

st.markdown("### Have a dedicated page for the chatbot")

st.code("""
#st.session_state.setdefault('user_message', [])
url_params = st.experimental_get_query_params() 

if 'pdf_index' in url_params:
    st.session_state['pdf_index'] = url_params['pdf_index'][0]
""", language="python")

st.code("""
if "pdf_index" not in st.session_state:
    #➡️
    #file_path = st.file_uploader(label="Upload a PDF file that your chatbot will use", type=['pdf'])

    #if st.button("Index PDF", disabled=not bool(file_path)):
#...
else:
    #➡️
    # Define Streamlit Containers
    #response_container = st.container()
""", language="python")

st.code("""
#st.session_state['pdf_index'] = pdf_index

# Direct user to the link to chat
st.markdown(f"PDF indexed successfully as **{st.session_state.pdf_index}**. The app to chat with your document can be found here: [http://localhost:8501?pdf_index={st.session_state.pdf_index}](http://localhost:8501?pdf_index={st.session_state.pdf_index}).")
""", language="python")