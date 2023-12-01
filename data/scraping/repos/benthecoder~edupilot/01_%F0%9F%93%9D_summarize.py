import streamlit as st
from utils import generate_word_document
from llama_index import SimpleDirectoryReader
import pathlib
from llama_index.llms import OpenAI
from llama_index import ServiceContext
from llama_index.response_synthesizers import Refine


PROJECT_DIR = pathlib.Path(__file__).parent.parent

st.title("Transcript Summarizer")

content = ""
with st.form("my_form"):
    transcript_file = st.selectbox(
        "Select a transcript file", options=st.session_state.transcripts
    )

    # Submit button
    submitted = st.form_submit_button("Summarize Transcript")
    if submitted:
        input_file = PROJECT_DIR / "transcripts" / transcript_file
        reader = SimpleDirectoryReader(input_files=[input_file])
        docs = reader.load_data()
        text = docs[0].text

        llm = OpenAI(model="gpt-4")
        service_context = ServiceContext.from_defaults(llm=llm)
        summarizer = Refine(service_context=service_context, verbose=True)
        response = summarizer.get_response(
            "Summarize this lecture using the cornell system", [text]
        )

        st.write(response)


if content != "":
    doc_file = generate_word_document(content)
    st.download_button(
        label="Download summary",
        data=doc_file,
        file_name="assignment.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
