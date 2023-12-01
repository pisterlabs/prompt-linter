import logging
import tiktoken
import gradio as gr
from langchain.text_splitter import CharacterTextSplitter
from utils import json_validator, fetch_chat
from typing import List


class Reviwer:
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
            The following is my research paper:
            ```
            {text}
            ```

            This paper has been submitted.
            You are the reviwer that has to make sure this paper meets the high standard.
            Do not judge the paper by its own claims but by your own knowledgeable insight in the research field.
            Please provide the review in the following json format:
            {{
                "summary": "Summarize the paper's main contribution and significance.",
                "strength": "List three, or more, strong aspects of this paper. Please number each point.",
                "weakness": "List three, or more, weak aspects of this paper. Please number each point.",
                "comments": "Detailed comments to the authors. Point out the problems in the paper.",
                "Overall Recommendation": ["strong reject", "weak reject", "board line", "weak accept", "accept", "strong accept"]
                "confidence": "Your confidence level 1-5 about the research field to justify for your overall recommendation.",
                "familiarity": "Describe your confidence level."
            }}
            """
            return json_validator(fetch_chat(prompt, openai_key, model=self.model))
        except Exception as e:
            raise gr.Error(str(e))
