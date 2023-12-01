from fastapi import APIRouter, File, UploadFile, Form
import pdfplumber
from app.api.utils import get_token_len, break_up_tokens, detokenize
from nltk import word_tokenize
from io import BytesIO
import openai
import tiktoken
from dotenv import load_dotenv
import os
import time
from app.api.models import example_summary
from app.api.utils import to_text_from_pdf
from app.api.utils import summarize_by_chatgpt, query_doc_content
from typing import List
from langchain.memory import ChatMessageHistory

router = APIRouter()
load_dotenv()
openai_key = os.getenv("OPENAI_KEY")

@router.post("/summarize")
async def summarize(file: UploadFile = File(...), questions: List[str] = Form(...)):
    """
    Endpoint that accepts a PDF file via a POST request.
    """

    print(questions)

    openai.api_key = openai_key

    file_in_memory = BytesIO(await file.read())
    complete_text = " ".join(example_summary) # to_text_from_pdf(file_in_memory)

    summary = summarize_by_chatgpt(complete_text)
    q_a_pairs = [(question, query_doc_content(question)) for question in questions]
    
    return {
        "filename": file.filename,
        "text": summary,
        "q-a-pairs": q_a_pairs
    }
