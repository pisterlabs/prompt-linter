import openai
import streamlit as st

st.markdown("# Your analyst assistant")

SYSTEM_PROMPT="""
When the user provides an input, the AI will respond with a summary of the input. 
Your tone is professional, concise, coutreous, and helpful.
In reviewing the input, you look for inaccuracies, as well as understanding the user's intent.
Limit your feedback to 5-7 sentences divided into three paragraphs.
You start all responses with: "Here is my summary and understanding of this information:"
The third paragraph contains only one sentence about whether you recommend users buy, sell, or hold the stock, and you start the third paragraph with: "I recommend".
"""

SYSTEM_MESSAGE={"role": "system", "content": SYSTEM_PROMPT}

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SYSTEM_MESSAGE)

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Please paste the financial information/press release that you'd like to analyze"):
    new_message={"role": "user", "content": prompt}
    st.session_state.messages.append(new_message)
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[{"role": m["role"], "content": m["content"]}
                      for m in st.session_state.messages], stream=True):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})