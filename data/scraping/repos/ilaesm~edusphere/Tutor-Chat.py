import openai
import streamlit as st

st.set_page_config(layout='wide', page_title="Virtual", page_icon=":sleep:")
col1, col2 = st.columns(2)
with col1:
    with open("logo.svg", "r") as file:
        svg_logo = file.read()
    st.markdown(svg_logo, unsafe_allow_html=True)

st.divider()
st.subheader("Hi, I am the EduAI Virtual, I can help you with anything, from basic math to motivation")
openai.api_key = st.sidebar.text_input('Please enter your OpenAI API Key', type="password")
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hi"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})