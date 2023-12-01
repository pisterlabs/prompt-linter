import streamlit as st
import os 
import openai
import pandas as pd
from streamlit_chat import message

st.title('Patient Co-Pilot 🩺')
st.write('Patient Co-Pilot is a chatbot that supports patients with their upcoming surgeries. It can answer any questions regarding pre-operative and post-operative care.')

openai.api_key = os.environ.get('OPENAI-KEY')

CONTENT = open('resources/system_prompt.txt', 'r').read()

def get_response():
    message_placeholder = st.empty()
    full_response = ""
    messages = [{"role": "system", "content": CONTENT}] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    if "surgery" in st.session_state:
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages= messages,
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
    else:
        st.session_state["surgery"] = messages[-1]['content']
        full_response = f'Thank you! You replied {st.session_state["surgery"]}. Do you have questions about this surgery?'
    message_placeholder.markdown(full_response)
    return full_response

def main():
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    # load previous messages, or empty list if there are no previous messages
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content":"Hi! I'm Patient Co-Pilot. What is your surgery?"}]



    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # chatting part
    if prompt := st.chat_input("How can I help you?"):
        # user input
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # response
        with st.chat_message("assistant"):
            full_response = get_response()
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()