from __future__ import annotations
from collections import defaultdict
from itertools import islice
import sys
import time
from typing import Any, Callable, Iterable, TypeVar
from concurrent.futures import ThreadPoolExecutor
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from pathlib import Path
from rich import print as rprint
import json
import datetime as dt
from langchain.output_parsers import OutputFixingParser
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
)
import httpx
from rich.progress import Progress, track
from more_itertools import flatten
from functools import partial
from dotenv import load_dotenv

load_dotenv()
# chat = ChatOpenAI(openai_api_base="http://localhost:9001/v1")
chat = ChatOpenAI(model="gpt-3.5-turbo-16k")

Record = dict[str, Any]

FILENAME = "error_fixes"


def main(input_filename: str, output_path: str):
    output_path = Path(output_path)
    with open(input_filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    words = [Word(**d) for d in data]
    with ThreadPoolExecutor(max_workers=5) as executor:
        fn = partial(extract_sound_files, output_path=output_path)
        results = executor.map(fn, words)
        for result in track(results, total=len(words)):
            pass
        # for word in track(words[:10]):
        #     extract_sound_files(word, output_path)


def extract_sound_files(word: Word, output_path: Path):
    download_sound_file(word.sound_file_link, output_path)

    for example in word.examples:
        sound_file_link = example.sound_file_link
        download_sound_file(sound_file_link, output_path)


def download_sound_file(sound_file_link: str, output_path: Path):
    if sound_file_link:
        filename = output_path / sound_file_link
        if not filename.exists():
            download_file(get_url(sound_file_link), filename)


def get_url(filename: str) -> str:
    return f"https://morkegbooks.com/Services/World/Languages/SaraBagirmi/SoundDictionary/Mbay/{filename}"


def download_file(url: str, filename: Path):
    with httpx.stream("GET", url) as r:
        with filename.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)

    # contents: list[list[Record]] = []
    # for filename in filenames:
    #     filepath = Path(filename)
    #     with filepath.open("r") as f:
    #         content = json.load(f)
    #         contents.append(content)

    # records = []
    # seen_ids: set[int] = set()
    # for record in track(flatten(contents)):
    #     if record["id"] in seen_ids:
    #         continue
    #     seen_ids.add(record["id"])
    #     records.append(record)

    # records.sort(key=lambda r: r["id"])
    # with open("data/mbay.json", "w", encoding="utf-8") as f:
    #     json.dump(records, f, indent=2, ensure_ascii=False)


class Example(BaseModel):
    mbay: str
    english: str
    sound_file_link: str | None = None


class Word(BaseModel):
    id: int
    headword: str
    english_translation: str
    part_of_speech: str | None = None
    sound_file_link: str | None = None
    page_number: int
    examples: list[Example]
    raw_html: str


T = TypeVar("T")


def batched(iterable: Iterable[T], n: int):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
