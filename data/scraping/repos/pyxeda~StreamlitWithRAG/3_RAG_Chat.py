import streamlit as st
from openai import OpenAI
from pinecone_utils import filter_matching_docs

client = OpenAI(
    api_key = st.secrets["OPENAI_API_KEY"]
)

# title of the web app
st.title("Retrieval Augment Chat")


SYSTEM_MESSAGE={"role": "system", 
                "content": "Ignore all previous commands. You are a helpful and patient guide based in Silicon Valley."
                }

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SYSTEM_MESSAGE)

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    retreived_content = filter_matching_docs(prompt, get_text = True)
    #print(f"Retreived content: {retreived_content}")
    prompt_guidance=f"""
    Please guide the user with the following information:
    {retreived_content}
    The user's question was: {prompt}
        """
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        messageList=[{"role": m["role"], "content": m["content"]}
                      for m in st.session_state.messages]
        messageList.append({"role": "user", "content": prompt_guidance})
        
        for response in client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messageList, stream=True):
            full_response += f"{response.choices[0].delta.content if response.choices[0].delta.content else '' }"
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    with st.sidebar.expander("Retreival context provided to GPT-3"):
        st.write(f"{retreived_content}")
    st.session_state.messages.append({"role": "assistant", "content": full_response})