import os
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# app.py (continue)

def on_submit_button_click():
    st.toast("Processing... Please wait...", icon='⏳')

    # Prepare the system message
    message_system = SystemMessage(content="You're are a helpful," 
                                          "talkative, and friendly assistant.")
    
    # Prepare the user message using the value from the `text_area`
    message_user = HumanMessage(content=st.session_state.input_text)

    full_response = []
    # Loop through the chunks streamed back from the API call
    for resp in model.stream([message_system, message_user]):
        wordstream = resp.dict().get('content')

        # if wordstream is not None
        if wordstream:
            full_response.append(wordstream)
            result = "".join(full_response).strip()
            # This streaming_box is a st.empty from the display
            with streaming_box.container():
                st.markdown('---')
                st.markdown('#### Response:')
                st.markdown(result)
                st.markdown('---')

    # Concatenate and store the streamed chunks to a full response
    st.session_state.output_text = "".join(full_response).strip()

    st.toast("Processing complete!", icon='✅')


# main form
# This section is repeated here to show where the function
# `on_submit_button_click` above should be placed.
# This section of code been introduced earlier on.

model = ChatOpenAI(streaming=True,
                   callbacks=[StreamingStdOutCallbackHandler()],
                   verbose=True)

with st.form(key='form_main'):
    user_input = st.text_area("Enter your text here", key='input_text')

    submit_button = st.form_submit_button(
                            label='Submit', 
                            on_click=on_submit_button_click)
    
streaming_box = st.empty()

# For showing the Completed Output
if 'output_text' in st.session_state:
    st.markdown('---')
    st.markdown('#### Response:')
    st.markdown(st.session_state['output_text'])
    st.markdown('---')