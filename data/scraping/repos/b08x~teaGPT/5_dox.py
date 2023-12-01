import sys, os
import streamlit as st

import openai

try:
    if os.environ["OPENAI_API_KEY"]:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    else:
        openai.api_key = st.secrets.OPENAI_API_KEY
except Exception as e:
    st.write(e)

dox_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
+ '/pages/dox')

sys.path.append(dox_dir)

print(sys.path)

from sidebar import sidebar

from ui import (
    wrap_doc_in_html,
    is_query_valid,
    is_file_valid,
    is_open_ai_key_valid,
    display_file_read_error,
)

from core.caching import bootstrap_caching

from core.parsing import read_file
from core.chunking import chunk_file
from core.embedding import embed_files
from core.qa import query_folder

EMBEDDING = "openai"
VECTOR_STORE = "faiss"
MODEL = "openai"

# For testing
# EMBEDDING, VECTOR_STORE, MODEL = ["debug"] * 3

st.set_page_config(page_title="KnowledgeGPT", page_icon="📖", layout="wide")
st.header("📖KnowledgeGPT")

# Enable caching for expensive functions
bootstrap_caching()

sidebar()


uploaded_file = st.file_uploader(
    "Upload a pdf, docx, or txt file",
    type=["pdf", "docx", "txt"],
    help="Scanned documents are not supported yet!",
)

if not uploaded_file:
    st.stop()

try:
    file = read_file(uploaded_file)
except Exception as e:
    display_file_read_error(e)

chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)

if not is_file_valid(file):
    st.stop()

if not is_open_ai_key_valid(openai.api_key):
    st.stop()


with st.spinner("Indexing document... This may take a while⏳"):
    folder_index = embed_files(
        files=[chunked_file],
        embedding=EMBEDDING,
        vector_store=VECTOR_STORE,
        openai_api_key=openai.api_key,
    )

with st.form(key="qa_form"):
    query = st.text_area("Ask a question about the document")
    submit = st.form_submit_button("Submit")


with st.expander("Advanced Options"):
    return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
    show_full_doc = st.checkbox("Show parsed contents of the document")


if show_full_doc:
    with st.expander("Document"):
        # Hack to get around st.markdown rendering LaTeX
        st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)


if submit:
    if not is_query_valid(query):
        st.stop()

    # Output Columns
    answer_col, sources_col = st.columns(2)

    result = query_folder(
        folder_index=folder_index,
        query=query,
        return_all=return_all_chunks,
        model=MODEL,
        openai_api_key=openai.api_key,
        temperature=0,
    )

    with answer_col:
        st.markdown("#### Answer")
        st.markdown(result.answer)

    with sources_col:
        st.markdown("#### Sources")
        for source in result.sources:
            st.markdown(source.page_content)
            st.markdown(source.metadata["source"])
            st.markdown("---")
