from typing import NoReturn

import streamlit as st

from langchain_lab import logger


def display_error(e: Exception) -> NoReturn:
    st.error(e)
    logger.error(f"{e.__class__.__name__}: {e}")
    st.stop()
