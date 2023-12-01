import openai
import streamlit as st
import os

# set variables
openai.api_key = os.environ["OPENAI_API_KEY"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

# page title
st.title("ChatGPT-like clone")

# lets have a sidebar of controls
with st.sidebar:
    chat_type = st.radio(
        "Select a conversation type",
        ("Python", "SQL", "Data Science", "Role Playing")
    )

with st.sidebar:
    temperature = st.slider(
        label = "Creativity (Temperature)",
        min_value = 0.0,
        max_value = 1.0,
        value = 0.2,
        step = 0.1,
    )

with st.sidebar:
    top_p= st.slider(
        label = "Vocabulary Size (top_p)",
        min_value = 0.0,
        max_value = 1.0,
        value = 0.1,
        step = 0.1,
    )

with st.sidebar:
    query_memory = st.slider(
        label = "Query Memory",
        min_value = 1,
        max_value = 20,
        value = 8,
        step = 1,
    )

# lets use our radio button inputs to create instructions for our different response modes
system_prompts = {
    "Python": " ".join(
        [
            "You are an expert specialising in programming in Python.",
            "Please ensure that your answers are correct and relevant.",
            "When you produce sample code, please ensure the code is syntactically, and logically correct Python 3.",
            "Please respond to the following chat:",
        ]
    ),
    "SQL": " ".join(
        [
            "You are an expert specialising in programming in Transact-SQL.",
            "Please ensure that your answers are correct and relevant.",
            "When you produce sample code, please ensure the code is syntactically, and logically correct TSQL.",
            "Please respond to the following chat:",
        ]
    ),
    "Data Science": " ".join(
        [
            "You are an expert specialising in Data Science.",
            "You are well veresed in Python, PySpark, pandas, matplotlib, seaborn, scikit-learn, Pytorch and Tensorflow.",
            "Please ensure that your answers are correct and relevant.",
            "When you produce suggestions, please explain the reasoning behind your choices.",
            "If asked to provide code, please assume you should produce Python 3 code."
            "Please respond to the following chat:",
        ]
    ),
    "Role Playing": " ".join(
        [
            "You are an expert specialising in Role Playing games.",
            "Please ensure that your answers are correct and relevant.",
            "When you make suggestions for characters, NPCs, items, or places, please make them interesting and unique.",
            "Please respond to the following chat:",
        ]
    ),
}

# now lets set up our chat bot

# print all the messages to date
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# take our next input and process
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {
                    "role": message["role"], 
                    "content": message["content"],
                }
                # lets only send the more recent to the chatbot to save on processing cost
                for message in st.session_state.messages[-(2*query_memory):-1]
            ]+[
                {
                    "role": "system",
                    # set the appropriate system prompt using the current radio button value
                    "content": system_prompts[chat_type],
                }
            ]+[
                {
                    "role": message["role"], 
                    "content": message["content"],
                }
                # finally send the most recent responses
                for message in st.session_state.messages[-1:]
            ],
            stream=True,
            temperature = temperature,
            top_p = top_p,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
