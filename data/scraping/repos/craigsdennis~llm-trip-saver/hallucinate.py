from dotenv import load_dotenv

load_dotenv()

import json
import os
import requests
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)


@st.cache_resource
def get_chat():
    return ChatOpenAI(
        model_name=os.environ["OPENAI_MODEL"], temperature=0.7, max_tokens=None
    )


@st.cache_data
def get_available_companies():
    filenames = os.listdir(os.path.join("trips", "companies"))
    return sorted([file.split(".")[0] for file in filenames])


chat = get_chat()

"""
# Trip Saver

There are cases where you might want to force an AI hallucination and then save it
for replay to invite others in to the same context.
"""


def load_trip_from_filename(trip_path):
    full_filepath = os.path.join(
        trip_path, st.session_state["trip_file_name"] + ".json"
    )
    with open(full_filepath) as file:
        json_str = file.read()
    import_json(json.loads(json_str))


with st.form("load_existing"):
    trip_file_name = st.selectbox(
        "Choose an existing Company Trip",
        options=get_available_companies(),
        format_func=lambda x: x.title(),
        key="trip_file_name",
    )
    st.form_submit_button(
        "Load",
        on_click=load_trip_from_filename,
        kwargs={"trip_path": os.path.join("trips", "companies")},
    )


"""
## System prompt

This is how you'd like the system to behave. You should be very specific, and make sure
that you tell it is okay to provide fake information. After you set this up, you can talk 
directly to it.
"""

if "system_locked" not in st.session_state:
    st.session_state["system_locked"] = False

system_prompter = st.text_area(
    "How should this assistant behave?",
    key="system_prompter",
    disabled=st.session_state["system_locked"],
)

"""
## Interact

Now you should interact with the assistant you just created.
"""

if "history" not in st.session_state:
    st.session_state["history"] = []
else:
    st.markdown("### History")


def remove_response_and_prompt(response_index):
    prompt_index = response_index - 1
    prompt = st.session_state["history"].pop(prompt_index)
    print(f"Removing prompt {prompt}")
    # Note since the list has shifted up we are using the same index
    ai = st.session_state["history"].pop(prompt_index)
    print(f"Removing ai result: {ai}")


for index, message in enumerate(st.session_state["history"]):
    if isinstance(message, HumanMessage):
        st.markdown(f"#### {message.content}")
    elif isinstance(message, AIMessage):
        st.markdown(message.content)
        st.button(
            "Remove this prompt",
            key=f"remove_{index}",
            on_click=remove_response_and_prompt,
            args=(index,),
        )


if st.session_state["history"]:
    number_of_tokens = chat.get_num_tokens_from_messages(st.session_state["history"])
    st.markdown(
        f"**Number of Tokens used in current conversation**: {number_of_tokens}"
    )


def submit_chat():
    messages = st.session_state["history"]
    if not messages:
        messages.append(SystemMessage(content=system_prompter))
    messages.append(HumanMessage(content=st.session_state.chat_prompter))
    try:
        ai_message = chat(messages)
    except Exception as ex:
        ai_message = AIMessage(
            content=f"Hmmm...something went wrong. Try again with a different prompt. {ex}"
        )
    messages.append(ai_message)
    st.session_state["history"] = messages
    st.session_state["system_locked"] = True
    st.session_state.chat_prompter = ""


with st.form("chat_prompt"):
    chat_prompter = st.text_area(
        "What would you like to ask your assistant?", key="chat_prompter"
    )
    st.form_submit_button("Ask", on_click=submit_chat)


def import_json(json_obj):
    messages = messages_from_dict(json_obj)
    st.session_state["history"] = messages
    st.session_state["system_prompter"] = messages[0].content
    st.session_state["system_locked"] = True
    st.balloons()


def import_json_url():
    url = st.session_state["json_url"]
    response = requests.get(url)
    json = response.json()
    return import_json(json)


with st.expander("Additional storage and retrieval options"):
    with st.form("save_trip"):
        name = st.text_input("What should this be called?")
        submitted = st.form_submit_button("Save")
        if submitted:
            filepath = os.path.join("trips", "user-submitted", name + ".json")
            with open(filepath, "w") as f:
                f.write(json.dumps(messages_to_dict(st.session_state["history"])))
            print(f"Saved: {filepath}")

    with st.form("import_json_url"):
        json_url = st.text_input("Import from JSON URL", key="json_url")
        st.form_submit_button("Import", on_click=import_json_url)

    st.markdown("#### Raw JSON")
    st.code(json.dumps(messages_to_dict(st.session_state["history"])))
