import streamlit as st
from openai_utils import get_response


SYSTEM_MESSAGE={"role": "system", 
                "content": "Ignore all previous commands. You are a helpful and patient guide."
                }

# session states
if 'prompt' not in st.session_state:
    st.session_state.prompt = []
    st.session_state.prompt.append(SYSTEM_MESSAGE)

# set title
st.title("Message with Prompt Engineering")

# text input
text = st.text_input("Set a Prompt", help = "Prompt Engineering")

if text:
    st.session_state.prompt.append({
        "role": "user",
        "content": f"{text}"
    })

    with st.status("Getting the response...."):
        # get the response
        response = get_response(st.session_state.prompt)
        st.session_state.prompt.append({
            "role": "system",
            "content": f"{response}"
        })
        st.write("Response: {}".format(response))
