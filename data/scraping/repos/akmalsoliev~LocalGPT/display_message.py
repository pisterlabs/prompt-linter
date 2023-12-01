import os 
import streamlit as st 
from langchain.schema import (
    AIMessage,
    HumanMessage
)
import markdown2
import pdfkit
import pyperclip

def display_messages():
    for index, message in enumerate(st.session_state.messages):
        if type(message) in [AIMessage]:
            with st.chat_message("assistant"):
                st.write(message.content)

            if index > 1:
                col1, col2, col3 = st.columns(3, gap="small")

                with col1:
                    if st.button("Export Markdown", key=f"emd-{index}"):
                        name = message.content[:30]
                        file_name = os.path.join("io", "markdown", f"{name}.md")
                        with open(file_name, "w") as f:
                            f.write(message.content)
                        st.write("Export Successfully!")

                with col2:
                    if st.button("Export PDF", key=f"epdf-{index}"):
                        name = message.content[:30]
                        md_content = markdown2.markdown(message.content)
                        file_name = os.path.join("io", "pdf", f"{name}.pdf")
                        pdfkit.from_string(md_content, file_name)
                        st.write("Export Successfully!")

                with col3:
                    if st.button("Copy", key=f"copy-{index}"):
                        pyperclip.copy(message.content)

        elif type(message) == HumanMessage:
            with st.chat_message("user"):
                st.write(message.content)
