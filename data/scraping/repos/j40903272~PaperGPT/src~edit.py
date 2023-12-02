import logging
import tiktoken
import gradio as gr
from langchain.text_splitter import CharacterTextSplitter
from utils import fetch_chat
from typing import List


class Editor:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.encoder = tiktoken.encoding_for_model(model)
        self.model = model
        with open("./sample/sample_abstract.tex", "r") as f:
            self.sample_content = f.read()

    def split_chunk(self, text, chunk_size: int = 2000) -> List[str]:
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=0
        )
        text_list = text_splitter.split_text(text)
        return text_list

    def generate(self, text: str, openai_key: str):
        logging.info("start editing")

        try:
            prompt = f"""
            I am a computer science student.
            I am writing my research paper.
            You are my editor.
            Your goal is to improve my paper quality at your best.
            Please edit the following paragraph and return the modified paragraph.
            If the paragraph is written in latex, return the modified paragraph in latex.

            ```
            {text}
            ```
            """
            return fetch_chat(prompt, openai_key, model=self.model)
        except Exception as e:
            raise gr.Error(str(e))
