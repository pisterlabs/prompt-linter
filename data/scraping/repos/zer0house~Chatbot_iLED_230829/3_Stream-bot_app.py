import openai
import streamlit as st

st.title("Stream-Bot")

openai.api_key = st.secrets["OPENAI_API_KEY"]

if "openai_model" not in st.session_state:
    st.session_state.openai_model = "gpt-3.5-turbo"


with st.chat_message(name="assistant", avatar="https://avatars.githubusercontent.com/u/78703832?v=4"):
    st.write("Hello 😀")
    st.write("I'm GPT Bot. I can answer your questions about GPT-3.5 Turbo.")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        

prompt = st.chat_input("What is up?")
if prompt: 
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    # stream 기능 추가
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # 계속 만들어지고 있을 동안 반복
        for response in openai.ChatCompletion.create(
            model=st.session_state.openai_model,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages 
            ],
            max_tokens=256,
            temperature=1,
            top_p=0.5, 
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "...") # 작성 중일 때 위에 ... 추가
        # 다 만들어지면
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
