import streamlit as st
import ai_parrot.v1_brain as v1
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()

im = Image.open("app/calvin.ico")

st.set_page_config(
    page_title="Calvinist Parrot v1", 
    page_icon="🦜",
    layout="wide",
    menu_items={
        'Get help': 'https://svrbc.org/',
        'About': "v2.2\n\nCreated by: [Jesús Mancilla](mailto:jgmancilla@svrbc.org)\n\nFrom [SVRBC](https://svrbc.org/)\n\n"
    }
)

def reset_status():
    st.session_state["messages"] = [{"role": "parrot", "avatar": "🦜", "content": "What theological questions do you have?"}]
    st.session_state["parrot_conversation_history"] = [{"role": "system", "content": v1.parrot_sys_message}]
    st.session_state["calvin_conversation_history"] = [{"role": "system", "content": v1.calvin_sys_message}]

def update_status(msg):
    st.session_state["messages"].append(msg)
    if msg['role'] == "parrot":
        st.session_state["parrot_conversation_history"].append({"role": "system", "content": msg["content"]})
        st.session_state["calvin_conversation_history"].append({"role": "system", "content": f'/parrot/ {msg["content"]}'})
    else:
        st.session_state["parrot_conversation_history"].append({"role": "system", "content": f'/calvin/ {msg["content"]}'})
        st.session_state["calvin_conversation_history"].append({"role": "system", "content": msg["content"]})

def interactWithAgents(question):
    st.session_state["parrot_conversation_history"].append({"role": "user", "content": f'/human/ {question}'})
    st.session_state["calvin_conversation_history"].append({"role": "user", "content": f'/human/ {question}'})
    
    with st.chat_message("parrot", avatar="🦜"):
        answer = ''
        c = st.empty()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=st.session_state["parrot_conversation_history"],
            stream=True,
            temperature = 0
        )
        for event in response:
            c.write(answer.split('/')[-1])
            event_text = event.choices[0].delta.content
            if event_text is not None:
                answer += event_text

    update_status({"role": "parrot", "avatar": "🦜", "content": answer.split('/')[-1]})

    with st.chat_message("calvin", avatar=im):
        answer = ''
        c = st.empty()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=st.session_state["calvin_conversation_history"],
            stream=True,
            temperature = 0
        )
        for event in response:
            c.write(answer.split('/')[-1])
            event_text = event.choices[0].delta.content
            if event_text is not None:
                answer += event_text

    update_status({"role": "calvin", "avatar": im, "content": answer.split('/')[-1]})

    with st.chat_message("parrot", avatar="🦜"):
        answer = ''
        c = st.empty()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=st.session_state["parrot_conversation_history"],
            stream=True,
            temperature = 0
        )
        for event in response:
            c.write(answer.split('/')[-1])
            event_text = event.choices[0].delta.content
            if event_text is not None:
                answer += event_text

    update_status({"role": "parrot", "avatar": "🦜", "content": answer.split('/')[-1]})

class main_parrot:
    def __init__(self):
        self.clear = st.sidebar.button("Reset chat history")
        st.sidebar.divider()

        # to show chat history on ui
        if "messages" not in st.session_state:
            reset_status()

    def main(self):
        if "page" not in st.session_state:
            st.session_state["page"] = "v1 Parrot"
        if st.session_state.page != "v1 Parrot":
            st.session_state["page"] = "v1 Parrot"
            reset_status()

        if self.clear:
            reset_status()

        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"], avatar=msg["avatar"]).write(msg["content"])

        if prompt := st.chat_input(placeholder="What is predestination?"):
            st.chat_message("user", avatar="🧑‍💻").write(prompt)
            st.session_state.messages.append({"role": "user", "avatar": "🧑‍💻", "content": prompt})
            interactWithAgents(prompt)


if __name__ == "__main__":
    obj = main_parrot()
    obj.main()