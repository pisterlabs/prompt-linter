from typing import List, NoReturn
import streamlit as st
from langchain.docstore.document import Document
from core.parser import File
import openai
from streamlit.logger import get_logger

logger = get_logger(__name__)

def wrap_doc_in_html(docs: List[Document]) -> str:
    text = [doc.page_content for doc in docs]
    if isinstance(text, list):
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])

def is_query_valid(query: str) -> bool:
    if not query:
        st.error("Soru yoksa, cevap da yok. Soruyu yazar mısın, lütfen?")
        return False
    return True

def is_file_valid(file: File) -> bool:
    if (
        len(file.docs) == 0
        or "".join([doc.page_content for doc in file.docs]).strip() == ""
    ):
        st.error("Belgeyi okuyamıyorum! Belgede metin olduğuna emin misin?")
        logger.error("Cannot read document")
        return False
    return True

def display_file_read_error(e: Exception) -> NoReturn:
    st.error("Belgeyi okuyamıyorum! Belgenin bozulmuş veya şifreli olmadığına emin olabilir misin?")
    logger.error(f"{e.__class__.__name__}: {e}")
    st.stop()

@st.cache_data(show_spinner=False)
def is_open_ai_key_valid(openai_api_key, model: str) -> bool:
    if model == "debug":
        return True
    
    if not openai_api_key:
        st.error("OpenAI API anahtarında bir sorun var 👀")
        logger.error("Error in OpenAI key")
        return False
    try:
        openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": "test"}],
            api_key=openai_api_key,
        )
    except Exception as e:
        st.error(f"{e.__class__.__name__}: {e}")
        logger.error(f"{e.__class__.__name__}: {e}")
        return False
    return True

