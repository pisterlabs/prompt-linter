import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler

from st_components.st_message import format_message


class CallbackHandler(BaseCallbackHandler):
    AVATAR_URL = "app/static/chatDuck.png"
    MESSAGE_ALIGNMENT = "flex-start"
    MESSAGE_BG_COLOR = "#71797E"
    AVATAR_CLASS = "bot-avatar"

    def __init__(self):
        self.token_buffer = []
        self.placeholder = None
        self.has_streaming_ended = False

    @staticmethod
    def _create_message_div(
        text, avatar_url, message_alignment, message_bg_color, avatar_class
    ):
        formatted_text = format_message(text)
        return f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: {message_alignment};">
                    <img src="{avatar_url}" class="{avatar_class}" alt="avatar" style="width: 50px; height: 50px;" />
                    <div style="background: {message_bg_color}; color: white; border-radius: 20px; padding: 10px; margin-right: 5px; max-width: 75%; font-size: 14px; font-family: ui-monospace;">
                        {formatted_text} \n </div>
                </div>
            """

    def on_llm_new_token(self, token, run_id, parent_run_id=None, **kwargs):
        self.token_buffer.append(token)
        complete_message = "".join(self.token_buffer)
        container_content = self._create_message_div(
            complete_message,
            self.AVATAR_URL,
            self.MESSAGE_ALIGNMENT,
            self.MESSAGE_BG_COLOR,
            self.AVATAR_CLASS,
        )
        if self.placeholder is None:
            self.placeholder = st.markdown(container_content, unsafe_allow_html=True)
        else:
            self.placeholder.markdown(container_content, unsafe_allow_html=True)

    @staticmethod
    def display_dataframe(df):
        avatar_url = "https://pbs.twimg.com/profile_images/1274363897676521474/qgbqYYuV_400x400.jpg"
        message_alignment = "flex-start"
        avatar_class = "bot-avatar"

        st.write(
            f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: {message_alignment};">
                    <img src="{avatar_url}" class="{avatar_class}" alt="avatar" style="width: 50px; height: 50px;" />
                </div>
                """,
            unsafe_allow_html=True,
        )
        st.write(df)

    def on_llm_end(self, response, run_id, parent_run_id=None, **kwargs):
        self.token_buffer = []
        self.has_streaming_ended = True

    def __call__(self, *args, **kwargs):
        pass
