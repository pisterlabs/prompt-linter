import streamlit as st
from openai_utils import get_response

# web title
st.title("Message with AI")

# text input
text = st.text_input("Ask a Question")

if text:
    with st.status("Getting the response...."):
        convo = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{text}"}
        ]
        # get the response
        response = get_response(convo)
        st.write("Response: {}".format(response))
