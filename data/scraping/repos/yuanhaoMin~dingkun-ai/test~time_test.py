
import docx2txt
import os
import re
from fastapi import UploadFile
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Zilliz
from pdfminer.high_level import extract_text
from win32com import client
from win32com.client import constants


def process_docx_to_str(file: UploadFile) -> str:
    raw_text = docx2txt.process(file)
    processed_text = remove_blank_lines(raw_text)
    return processed_text


def process_pdf_to_str(file: UploadFile) -> str:
    raw_text = extract_text(file)
    processed_text = remove_blank_lines(raw_text)
    return processed_text


def remove_blank_lines(input_str: str) -> str:
    # Split the string into lines, filter out blank lines, and join the non-blank lines back into a string
    lines = input_str.splitlines()
    non_blank_lines = [line for line in lines if line.strip()]
    return "\n".join(non_blank_lines)


