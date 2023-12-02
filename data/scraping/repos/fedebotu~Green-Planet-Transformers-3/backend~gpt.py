import nltk
import pandas as pd
import os
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from typing import List
import numpy as np
import streamlit as st


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
@st.cache(suppress_st_warning=True, persist=True)
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


@st.cache(suppress_st_warning=True, persist=True)
def generate_answer(prompt: str, temperature=0) -> str:
    response = completion_with_backoff(
        model="text-davinci-002",
        prompt=prompt,
        temperature=temperature,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=[" Human:", " AI:"] #\n
    )
    return response["choices"][0]["text"].strip()