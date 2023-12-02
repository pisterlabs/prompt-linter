import os
import requests
from pathlib import Path
from urllib.parse import urlparse

pdf_url = 'https://cs229.stanford.edu/main_notes.pdf'
r = requests.get(pdf_url)
if r.status_code != 200:
    raise ValueError(
        "Check the url of your file; returned status code %s"
        % r.status_code
    )

# ファイル名を取得します
parsed_url = urlparse(pdf_url)
file_name = os.path.basename(parsed_url.path)

# pdf_filesディレクトリを作成し、ファイルを保存します
pdf_files_directory = Path("pdf_files")
pdf_files_directory.mkdir(exist_ok=True)
file_path = pdf_files_directory / file_name

with open(file_path, "wb") as f:
    f.write(r.content)

print(f"File saved at: {file_path}")

from dataclasses import dataclass
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from typing import List
import tiktoken


@dataclass
class Page:
    page_number: int
    text: str
    embedding: List[float]

def extract_page_text(pdf_file):
    pages_text = []
    for page_layout in extract_pages(pdf_file):
        single_page_text = ''
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                single_page_text += element.get_text().replace('\n', ' ')
        pages_text.append(single_page_text)
    return pages_text
file_path = f'pdf_files/{file_name}'
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
pages_text = extract_page_text(file_path)
# for page_text in pages_text:
#     print(page_text)
#     print(len(encoding.encode(page_text)))

from dotenv import load_dotenv
load_dotenv()
import openai
from openai.embeddings_utils import get_embedding
openai.api_key = os.getenv("OPENAI_API_KEY")
pages = []
for idx, page_text in enumerate(pages_text):
    # https://github.com/openai/openai-cookbook/blob/main/examples/Semantic_text_search_using_embeddings.ipynb
    embedding = get_embedding(
        page_text,
        engine="text-embedding-ada-002"
    )
    page = Page(page_number=idx + 1, text=page_text, embedding=embedding)
    pages.append(page)

import pickle
# ファイルにリストオブジェクトを保存する
with open(f'lecture_notes.pickle', 'wb') as f:
    pickle.dump(pages, f)