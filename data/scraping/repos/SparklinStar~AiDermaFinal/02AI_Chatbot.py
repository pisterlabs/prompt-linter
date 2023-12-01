from openai import OpenAI
import streamlit as st

st.title("DermaBot")

try:
    st.text("Disease Predicted: "+st.session_state["disease"])
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "user_input" not in st.session_state:
        st.session_state["user_input"] = "" # Initialize user_input

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state["user_input"] = prompt  # Store user input

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
        
        # Constructing the context within the messages
            user_input = st.session_state.get("disease", "")
        
            context_message = {
            "role": "system",
            "content": f"You are a skin disease diagnosis doctor. You have diagnosed {user_input} as a skin disease. Answer in short within 50 words. You are now explaining to the patient what {user_input} is. Do not say that you can not answer the question. Do not say I am an AI language model and not a doctor.",
            }
            messages_with_context = st.session_state.messages + [context_message]

            for response in client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in messages_with_context
                ],
                stream=True,
            ):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
        
            message_placeholder.markdown(full_response)
    
        st.session_state.messages.append({"role": "assistant", "content": full_response})

except KeyError:
    st.warning("Please upload disease image to generate pdf report.")

