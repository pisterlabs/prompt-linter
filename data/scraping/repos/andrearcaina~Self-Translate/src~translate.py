import cohere
from googletrans import Translator
import streamlit as st

cohere_client = cohere.Client(st.secrets["keys"]["COHERE_KEY"])

def translate_lang(text, target):
    response = cohere_client.detect_language(texts=[text])
    lang = response.results[0]

    if lang != target:
        result = Translator().translate(text, dest=target)
        text = result.text
    return text, lang.language_name
