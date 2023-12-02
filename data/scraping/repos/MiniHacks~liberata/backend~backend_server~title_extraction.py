import os
import openai
import tiktoken
import asyncio
from pydantic import BaseModel

try:
    openai.api_key = os.environ["OPENAI_API_KEY"]
except:
    import sys
    print("Could not find OPENAI_API_KEY -- check your env vars, and if you're having trouble, ask Ritik for help", file=sys.stderr)
    raise

class BookIdeaRow(BaseModel):
    og_text: str
    title: str
    author: str | None

    class Config:
        frozen = True

TOKEN_LIMIT = 2000
enc = tiktoken.get_encoding("gpt2")
async def extract_book_titles(text_block: str) -> list[BookIdeaRow]:
    tokens = enc.encode(text_block)
    print(type(tokens))
    i = 0
    
    tasks = []
    while i < len(tokens):
        tasks.append(asyncio.create_task(openai.Completion.acreate(
            model = "text-davinci-003",
            prompt = "Identify the book titles that are in the following block of text. "
                    "Do not provide book titles that are not mentioned in the text. "
                    "If you are not sure about who the author is,  write 'Unknown' in the table. "
                    "Provide the original snippet of text that made you recognize a book title. "
                    "Record every book you recognize, even if the title is not explicitly mentioned. "
                    # "Try to make the table as long as possible. "
                    "Use the following format:\n"
                    '"<original text 1>" || <book 1> || <author 1>\n'
                    '"<original text 2>" || <book 2> || <author 2>\n'
                    '...\n'
                    '\n'
                    "Text block:\n"
                    f"{enc.decode(tokens[i:i+TOKEN_LIMIT])}\n\n"
                    "Text || Title || Author\n"
                    "----------------------------\n",
            temperature = 0.76,
            max_tokens = 900
        )))

        i += TOKEN_LIMIT

    ret: list[BookIdeaRow] = []

    for task in tasks:
        result = await task
        out: str = result["choices"][0]["text"]
        print(f"\n\n\nllm out:\n{out}")
        books = [
            tuple(cell.strip() for cell in row.split("||"))
            for row in out.split("\n")
        ]
        for triple in books:
            if len(triple) == 3:
                og_text, title, author = triple
                if title != 'Unknown':
                    ret.append(BookIdeaRow(og_text = og_text[1:-1], title = title, author = author if author != "Unknown" else None))
            else:
                break
    return ret
