from collections import defaultdict

from typing import List, Literal
from itertools import islice
import sys
import time
from typing import Any, Iterable, TypeVar
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from pathlib import Path
from rich import print as rprint
import json
import datetime as dt
import langchain
from langchain.schema import SystemMessage
from langchain.output_parsers import OutputFixingParser
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
)
from rich.progress import track
from functools import partial
from dotenv import load_dotenv
from langchain.cache import SQLiteCache

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

load_dotenv()
chat = ChatOpenAI()

Record = dict[str, Any]


def main(filename: str):
    filepath = Path(filename)
    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)
    words = (Word(**d) for d in data)

    with ThreadPoolExecutor(max_workers=3) as executor:
        start = time.perf_counter()
        fn = partial(translate_record)
        tasks = [executor.submit(fn, word) for word in words]
        results = []
        batch_id = 0
        for i, task in enumerate(track(as_completed(tasks), total=len(tasks))):
            result = task.result()
            results.append(result)
            if i % 500 == 0:
                rprint(f"{dt.datetime.now()} - Creating checkpoint...", end="")
                save_results(
                    results,
                    filepath.parent / "checkpoints",
                    "mbay_dict_french",
                    f"_batch_{batch_id}",
                )
                batch_id += 1
                rprint("done")
        end = time.perf_counter()
        duration = dt.timedelta(seconds=end - start)
        rprint(f"{dt.datetime.now()} - Finished in {duration}")

    save_results(results, filepath.parent, "mbay_dict_french")


class Example(BaseModel):
    mbay: str
    english: str
    french: str | None = None
    sound_file_link: str | None = None


class Word(BaseModel):
    id: int
    headword: str
    english_translation: str
    french_translation: str | None = None
    part_of_speech: str | None = None
    sound_file_link: str | None = None
    page_number: int
    raw_html: str
    examples: List[Example]


class Translation(BaseModel):
    language: Literal["fr"] = Field("fr", description="The language of the translation")
    text: str = Field(..., description="the result of the translation")


class Result(BaseModel):
    status: Literal["success", "error"]
    result: Word | None = None
    error: dict[str, Any] | None = None


def save_results(results: Iterable[Result], dir: Path, filename: str, suffix: str = ""):
    groups = defaultdict(list)
    for r in results:
        if r.status == "success":
            groups["success"].append(r.result.dict())
        else:
            groups["error"].append(r.error)

    groups["success"].sort(key=lambda x: x["id"])
    outfilepath = dir / f"{filename}{suffix}.json"
    with outfilepath.open("w", encoding="utf-8") as f:
        json.dump(
            groups["success"],
            f,
            indent=2,
            ensure_ascii=False,
        )

    groups["error"].sort(key=lambda x: x["id"])
    error_filepath = dir / f"{filename}_errors{suffix}.json"
    with error_filepath.open("w", encoding="utf-8") as f:
        json.dump(groups["error"], f, indent=2, ensure_ascii=False)


def translate(
    mbay: str, english: str, part_of_speech: str | None = None
) -> Translation:
    _input = prompt.format(
        mbay_text=mbay,
        english_text=english,
        part_of_speech=part_of_speech,
        format_instructions=parser.get_format_instructions(),
    )
    output = chat(
        [_input, SystemMessage(content="Remember to format your output properly")]
    )
    result = parser.parse(output.content)
    return result.text


def translate_record(word: Word, verbose=False) -> Result:
    try:
        word.french_translation = translate(
            word.headword, word.english_translation, word.part_of_speech
        )
        for example in word.examples:
            example.french = translate(example.mbay, example.english)

    except Exception as e:
        return Result(
            status="error",
            error={
                "id": word.id,
                "word": word.dict(),
                "error": str(e),
                "type": e.__class__.__name__,
            },
        )

    return Result(status="success", result=word)


parser = OutputFixingParser.from_llm(
    parser=PydanticOutputParser(pydantic_object=Translation), llm=ChatOpenAI()
)

TRANSLATION_TMPL = """\
Translate this entry from a Mbay to English dictionary from English to French:
mbay text (for context): {{mbay_text}}
{% if part_of_speech %}
part of speech: {{part_of_speech}}
{% endif %}
english text: {{english_text}}

{{format_instructions}}"""


prompt = HumanMessagePromptTemplate.from_template(
    template=TRANSLATION_TMPL,
    input_variables=["mbay_text", "english_text", "part_of_speech"],
    template_format="jinja2",
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

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
    main(sys.argv[1])
