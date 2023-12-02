import openai
import streamlit as st

def main():
    st.title("🤖 Samuel GPT")
    st.caption("1.0 version by Samuel Lee (2023-07-06)")
    st.write("This is a smallll ChatGPT called Samuel GPT! Enter any thing you want to ask, but don't ask too much! (Cuz it costs me $$)")

    openai.api_key = st.secrets["OPENAI_API_KEY"]

    
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": "You are a helpful chatbot called Samuel GPT"}]

    for message in st.session_state.messages:
        if not message["role"] == "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    
    if prompt := st.chat_input("Write something"):
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
                max_tokens=200,
                temperature=0.75,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    main()