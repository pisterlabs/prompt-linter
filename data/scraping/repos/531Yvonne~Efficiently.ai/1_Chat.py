import streamlit as st
import openai
import csv
import json

st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI")

with st.sidebar:
    openai_api_key = st.text_input(
        "Add Your OpenAI API Key", key="chatbot_api_key", type="password")

    response_language = st.sidebar.selectbox(
        "Select Response Language", ["english", "french", "spanish", "chinese", "japanese", "korean"])

    # Export History Record
    # Format Selection
    export_format = st.selectbox(
        "Select Export Format", ["JSON", "CSV", "TXT"])

    # Export Chat History Button
    if st.sidebar.button("Export Chat History"):
        if export_format == "JSON":
            with open("chat_history.json", mode="w") as file:
                json.dump(st.session_state.messages, file, indent=4)
        elif export_format == "CSV":
            with open("chat_history.csv", mode="w", newline="") as file:
                writer = csv.writer(file)
                for msg in st.session_state.messages:
                    writer.writerow([msg["role"], msg["content"]])
        elif export_format == "TXT":
            with open("chat_history.txt", mode="w") as file:
                for msg in st.session_state.messages:
                    file.write(f"{msg['role']}: {msg['content']}\n")
        st.success(
            f"Chat history exported as {export_format} to the program file!")

    if st.button("Clear Chat History"):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi, I'm Yves. How can I help you?"}]
        st.success("Chat history cleared!")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not st.session_state["openai_api_key"]:
        st.info("Please add your OpenAI API key to continue.")
        # Maybe add a pop-up window here ?????????????
        st.stop()

    openai.api_key = st.session_state["openai_api_key"]
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg.content)
