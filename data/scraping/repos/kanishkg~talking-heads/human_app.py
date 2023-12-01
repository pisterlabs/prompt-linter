import csv
import random

import openai
import streamlit as st

DATA_FILE = "Pew_American_Trends_Panel_disagreement_500.csv"
MAX_MESSAGES = 6

# Preliminary setup
st.title("Change My View")
print("restarting")
print(st.session_state)
# st.write(st.session_state)

if "stage" not in st.session_state:
    st.session_state["stage"] = 1

print(f"session stage: {st.session_state['stage']}")

def get_data(i=None):
    with open(DATA_FILE, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        if i:
            data = list(reader)[i]
        else:
            data = random.choice(list(reader))
    question = data[3]
    answer_list = list(eval(data[4]))
    answer_list = answer_list[:-1] + ["Neutral"]
    random.shuffle(answer_list)
    return question, answer_list

# stage 4: thanks
if st.session_state["stage"] == 4:
    st.markdown("### Thanks for participating! You can close this tab now.")

# stage 3: final answer
if st.session_state["stage"] == 3:
    st.radio(
            st.session_state["question"],
            key="final_answer",
            options=st.session_state["answer_list"],
        )
    st.button("Submit", key="final", on_click=lambda: st.session_state.update({"stage": 4}))

# stage 2: conversation
if st.session_state["stage"] == 2:
    if "messages" not in st.session_state:
        st.session_state.messages = []

        message = f"When asked the question '{st.session_state['question']}' I feel the answer is '{st.session_state['initial_answer']}' but convince me that it is '{st.session_state['target_answer']}'."
        st.session_state.messages.append(
            {"role": "system", "content": "Convince the user of the opinion that they want to change to. Use rhetoric and critical arguments, dont merely list points. Be concise and respond to the arguments that the user makes. Make it more like a conversation than a list of points. Ask questions when required."}
        )
        st.session_state.messages.append(
            {"role": "user", "content": message}
        )

    for message in st.session_state.messages[2:]:
        with st.chat_message(message["role"]):
            print(message["role"])
            st.markdown(message["content"])

    if len(st.session_state.messages) == 2:
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

    if prompt:=st.chat_input("Type here to chat"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        if len(st.session_state.messages) > MAX_MESSAGES:
            suffix = "\nThis is the last message. Also provide a concluding remark with the response based on the discussion."
            st.session_state.messages[-1]["content"] += suffix  # add suffix to last message
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

    if len(st.session_state.messages) > MAX_MESSAGES+1:
        st.button("Next", key="next2", on_click=lambda: st.session_state.update({"stage": 3}))


# stage 1: get the question and answer
if st.session_state["stage"] == 1:
    st.text_input(label="OpenAI API Key", key="openai_api_key")
    if st.session_state["openai_api_key"]:
        openai.api_key = st.session_state["openai_api_key"]
    selected_model = st.selectbox(
        label="OpenAI Model",
        key="openaim",
        options=["gpt-4", "gpt-3.5-turbo"],
    )

    st.session_state["openai_model"] = selected_model
    print(st.session_state["openai_model"])

    if "question" not in st.session_state:
            st.session_state["question"], st.session_state["answer_list"] = get_data()
    # show the question and answer
    left_column, right_column = st.columns(2)

    with left_column:
        st.radio(
                st.session_state["question"],
                key="initial_answer",
                options=st.session_state["answer_list"],
            )

    with right_column:
        st.radio(
            "Target Answer",
            key="target_answer",
            options=st.session_state["answer_list"],
        )

    st.button("Next", key="next", on_click=lambda: st.session_state.update({"stage": 2}))