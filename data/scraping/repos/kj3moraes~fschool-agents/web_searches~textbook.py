import openai
import os
import PyPDF2
from utils import fill_prompt
from pathlib import Path
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from typing import List
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
client = Anthropic(
    api_key=CLAUDE_API_KEY
)


def read_pdf_with_sliding_window(pdf_path, window_size=20):
    """
    Read a PDF using a sliding window approach and yield text.

    Args:
        pdf_path (str): Path to the PDF file.
        window_size (int): Number of pages to read at a time.

    Yields:
        str: Text content of the PDF for each window.
    """
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(pdf_reader.pages) 

        for start_page in range(0, total_pages, window_size):
            end_page = min(start_page + window_size, total_pages)
            pdf_text = ""

            for page_num in range(start_page, end_page):
                page = pdf_reader.pages[page_num]
                pdf_text += page.extract_text()

            yield pdf_text


def retrieve_textbook_sections(textbook_path: Path, question: str) -> List:

    relevant_contexts = []
    for textbook_excerpt in read_pdf_with_sliding_window(textbook_path):
        query_prompt = fill_prompt(Path("./web_searches/prompts/textbook_extract.prompt"), 
                                    textbook=textbook_excerpt, 
                                    question=question)
        final_prompt = HUMAN_PROMPT + query_prompt + AI_PROMPT    
        completion = client.completions.create(
                    model="claude-2",
                    prompt=final_prompt,
                    max_tokens_to_sample=100000
            )
        relevant_contexts.append(completion.completion)
        print("Completed an excerpt search")
    return relevant_contexts 