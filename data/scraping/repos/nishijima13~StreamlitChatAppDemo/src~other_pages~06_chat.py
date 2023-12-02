# This page is for chat
import streamlit as st
from st_pages import add_page_title
import const
import datetime
import os
from PIL import Image
import openai
from streamlit_autorefresh import st_autorefresh
from modules import common
from modules.authenticator import common_auth
from modules.database import database

CHAT_ID = "0"
persona = None
llm = None
use_chatbot = False

CHATBOT_PERSONA = """
Please become a character of the following setting and have a conversation.

{persona}
"""

add_page_title()

authenticator = common_auth.get_authenticator()
db = database.Database()
if (
    common.check_if_exists_in_session(const.SESSION_INFO_AUTH_STATUS)
    and st.session_state[const.SESSION_INFO_AUTH_STATUS]
):
    messages = []
    # Check if chatbot is enabled
    tmp_use_chatbot = db.get_openai_settings_use_character()
    if tmp_use_chatbot == 1:
        persona = db.get_character_persona()[0]
        messages.append(
            {"role": "system", "content": CHATBOT_PERSONA.format(persona=persona)}
        )

        # Get chatbot settings
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            openai.api_key = openai_api_key
            persona = None
            st.error(
                "OPENAI_API_KEY is not set in the environment variables. Please contact the administrator."
            )

    user_infos = {}
    username = st.session_state[const.SESSION_INFO_USERNAME]
    name = st.session_state[const.SESSION_INFO_NAME]
    user_msg = st.chat_input("Enter your message")

    # Show old chat messages
    chat_log = db.get_chat_log(chat_id=CHAT_ID, limit=const.MAX_CHAT_LOGS)
    if chat_log is not None:
        for msg_info in chat_log:
            log_chat_id, log_username, log_name, log_message, log_sent_time = msg_info
            # Get user info
            if log_username not in user_infos:
                tmp_user_info = db.get_user_info(log_username)
                if tmp_user_info is None:
                    st.error(const.ERR_MSG_GET_USER_INFO)
                else:
                    user_infos[log_username] = {
                        "username": log_username,
                        "name": tmp_user_info[2],
                        "image_path": tmp_user_info[4],
                        "image": None,
                    }
            # Show chat message
            if log_username in user_infos:
                if (
                    user_infos[log_username]["image"] is None
                    and user_infos[log_username]["image_path"] is not None
                    and os.path.isfile(user_infos[log_username]["image_path"])
                ):
                    # Load user image
                    user_infos[log_username]["image"] = Image.open(
                        user_infos[log_username]["image_path"]
                    )
                with st.chat_message(
                    log_name, avatar=user_infos[log_username]["image"]
                ):
                    st.write(log_name + "> " + log_message)

                if persona is not None:
                    # Added conversation to give to chatbot.
                    if log_username == const.CHATBOT_USERNAME:
                        messages.append({"role": "assistant", "content": log_message})
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": log_name + " said " + log_message,
                            }
                        )
                    if len(messages) > const.MAX_CONVERSATION_BUFFER:
                        messages.pop(1)

    else:
        st.error(const.ERR_MSG_GET_CHAT_LOGS)

    # Show user message
    if user_msg:
        # Show new chat message
        db.insert_chat_log(
            chat_id=CHAT_ID,
            username=username,
            name=name,
            message=user_msg,
            sent_time=datetime.datetime.now(),
        )
        if username not in user_infos:
            # Get user info
            tmp_user_info = db.get_user_info(username)
            if tmp_user_info is None:
                st.error(const.ERR_MSG_GET_USER_INFO)
            else:
                user_infos[username] = {
                    "username": username,
                    "name": tmp_user_info[2],
                    "image_path": tmp_user_info[4],
                    "image": None,
                }
        if (
            username in user_infos
            and user_infos[username]["image"] is None
            and user_infos[username]["image_path"] is not None
            and os.path.isfile(user_infos[username]["image_path"])
        ):
            user_infos[username]["image"] = Image.open(
                user_infos[username]["image_path"]
            )
        with st.chat_message(name, avatar=user_infos[username]["image"]):
            st.write(name + "> " + user_msg)

        if persona is not None:
            # Show chatbot message
            messages.append({"role": "user", "content": name + " said " + user_msg})
            messages.append({"role": "assistant", "content": name + " said "})
            completion = openai.ChatCompletion.create(
                model=const.MODEL_NAME,
                messages=messages,
            )
            assistant_msg = completion["choices"][0]["message"]["content"]
            with st.chat_message(const.CHATBOT_NAME, avatar=const.CHATBOT_NAME):
                st.write(const.CHATBOT_NAME + "> " + assistant_msg)
            db.insert_chat_log(
                chat_id=CHAT_ID,
                username=const.CHATBOT_USERNAME,
                name=const.CHATBOT_NAME,
                message=assistant_msg,
                sent_time=datetime.datetime.now(),
            )

    # Refresh the page every (REFRESH_INTERVAL) seconds
    count = st_autorefresh(
        interval=const.REFRESH_INTERVAL, limit=None, key="fizzbuzzcounter"
    )
else:
    st.error("You are not logged in. Please go to the login page.")
