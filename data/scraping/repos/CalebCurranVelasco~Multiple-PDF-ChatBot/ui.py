# import streamlit as st
# import openai
# from streamlit.logger import get_logger

# logger = get_logger(__name__)

# @st.cache_data(show_spinner=False)
# def is_open_ai_key_valid(openai_api_key) -> bool:
#     if not openai_api_key:
#         st.error("Please enter your OpenAI API key in the sidebar!")
#         return False
#     try:
#         openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": "test"}],
#             api_key=openai_api_key,
#         )
#     except Exception as e:
#         st.error(f"{e.__class__.__name__}: {e}")
#         logger.error(f"{e.__class__.__name__}: {e}")
#         return False
#     return True