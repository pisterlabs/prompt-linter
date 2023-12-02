import streamlit as st
from utils import openai_call, generate_word_document

st.title("Transcript Summarizer")

content = ""
with st.form("my_form"):
    transcript_file = st.selectbox(
        "Select a transcript file", options=st.session_state.transcripts
    )

    with open("transcripts/" + transcript_file, "r") as f:
        lecture = f.read()

    # Submit button
    submitted = st.form_submit_button("Summarize Transcript")
    if submitted:
        st.session_state.messages.clear()

        # Create the prompt here
        prompt = f"""
        Consider this lecture transcript:
        {lecture}

        Write a high-quality summary of the lecture using the Cornell note taking method.
        """

        st.session_state.messages.extend(
            [
                {
                    "role": "system",
                    "content": "Your task is to summarize this lecture.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        message_placeholder = st.empty()
        content = openai_call(st.session_state.messages, message_placeholder)

if content != "":
    doc_file = generate_word_document(content)
    st.download_button(
        label="Download summary",
        data=doc_file,
        file_name="assignment.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
