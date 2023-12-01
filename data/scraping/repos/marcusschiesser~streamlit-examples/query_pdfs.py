import logging
import streamlit as st
from llama_index import (
    OpenAIEmbedding,
    ServiceContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.llms import OpenAI
from streamlit_examples.utils.theme import initPage

from streamlit_examples.utils.streamlit import cache_file, upload_files

initPage("QueryPDFs")
st.write(
    "Ask questions or create summaries or explanations on PDFs using [LlamaIndex](https://www.llamaindex.ai/)"
)


@st.cache_resource()
def get_service_context():
    llm = OpenAI(
        temperature=0.1, model="gpt-3.5-turbo", api_key=st.secrets["OPENAI_API_KEY"]
    )
    embed_model = OpenAIEmbedding()
    return ServiceContext.from_defaults(llm=llm, embed_model=embed_model)


@st.cache_data(show_spinner=False)
def query(filename, question):
    logging.info(f"Asking '{question}' on '{filename}'")
    documents = SimpleDirectoryReader(input_files=[filename]).load_data()
    index = VectorStoreIndex.from_documents(
        documents, service_context=get_service_context()
    )
    query_engine = index.as_query_engine()
    return query_engine.query(question)


def get_question():
    QUESTIONS = {
        "Summarize": "What is a summary of this document?",
        "Explain": "Explain this document",
    }

    mode = st.sidebar.selectbox("Select Mode", ("Summarize", "Explain", "Ask"))
    if mode == "Ask":
        question = st.sidebar.text_input("What's your question")
        if not question:
            st.sidebar.info("Please ask a question or select another mode.")
            st.stop()
    else:
        question = QUESTIONS[mode]
    return mode, question


mode, question = get_question()

# Upload PDFs
pdfs = upload_files(type="pdf", accept_multiple_files=True)

# Summarize each PDF
for pdf in pdfs:
    filename = cache_file(pdf, type="pdf")
    with st.spinner(f"{mode} '{pdf.name}'..."):
        summary = query(filename, question)
    with st.expander(f"'{pdf.name}'", expanded=True):
        st.markdown(summary)
