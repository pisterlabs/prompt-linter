import os
import uuid

import langchain
import requests
import server as st
from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import get_openai_callback
from langchain.schema import HumanMessage, AIMessage
from playsound import playsound
from streamlit_chat import message

from advisor.agents import init_convo_agent

langchain.debug = True


def init():
    load_dotenv(find_dotenv())

    st.set_page_config(
        page_title="Your Restaurant Advisor",
        page_icon="👩‍🍳",
    )
    st.header("Your Restaurant Advisor 👩‍🍳")

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def setup_agent():
    if 'agent' not in st.session_state:
        random_session_id = str(uuid.uuid4())
        st.session_state.agent = init_convo_agent(random_session_id)


def get_response_from_ai(human_input):
    setup_agent()
    print("="*20)
    with get_openai_callback() as cb:
        result = st.session_state.agent.run(human_input)
        print("Cost:", cb)
        return result


def get_voice_message(message):
    payload = {
        "text": message,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0,
        }
    }

    headers = {
        "accept": "audio/mpeg",
        "xi-api-key": os.getenv("ELEVEN_LABS_API_KEY"),
        "Content-Type": "application/json"
    }

    response = requests.post('https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM?optimize_streaming_latency=0', json=payload, headers=headers)
    if response.status_code == 200 and response.content:
        with open("audio.mp3", "wb") as f:
            f.write(response.content)
        playsound("audio.mp3")
        return response.content


def main():
    init()

    with st.sidebar:
        user_input = st.text_input("your message", value="")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Thinking..."):
            response = get_response_from_ai(user_input)
            # get_voice_message(response)
            st.session_state.messages.append(AIMessage(content=response))

    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages):
        if i % 2 == 0:
            message(msg.content, is_user=True, avatar_style="thumbs", key=str(i) + "_user")
        else:
            message(msg.content, is_user=False, avatar_style="avataaars", key=str(i) + "_ai")


if __name__ == "__main__":
    main()
