from collections import defaultdict

from typing import List, Literal
from itertools import islice
import sys
import time
import asyncio
from typing import Any, Iterable, TypeVar
from jinja2 import Environment
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from pathlib import Path
from rich import print as rprint
import json
import datetime as dt
from instructor import OpenAISchema, patch
from itertools import chain as iter_chain
from langchain.output_parsers import OutputFixingParser

from langchain.chat_models import ChatOpenAI
from rich.progress import track
from dotenv import load_dotenv
import openai

# langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

patch()
load_dotenv()

Record = dict[str, Any]

sem = asyncio.Semaphore(20)


async def main(base_filename: str, error_filename: str | None = None):
    filepath = Path(base_filename)
    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)
    words = (Word(**d) for d in data)

    if error_filename is not None:
        filepath = Path(error_filename)
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        words = iter_chain(words, (Word(**d["word"]) for d in data))

    start = time.perf_counter()
    tasks = [fix_record_translation(word, verbose=False) for word in words]
    results = []
    batch_id = 0

    for i, task in enumerate(track(asyncio.as_completed(tasks), total=len(tasks))):
        result = await task
        results.append(result)
        if i % 500 == 0:
            save_results(
                results,
                filepath.parent / "checkpoints",
                "mbay_dict_french_fixed",
                f"_batch_{batch_id}",
            )
            batch_id += 1
    end = time.perf_counter()
    duration = dt.timedelta(seconds=end - start)
    rprint(f"{dt.datetime.now()} - Finished in {duration}")

    save_results(results, filepath.parent, "mbay_dict_french_fixed")


FAILED_TRANSLATION_PHRASES = {"Translated text", None}
FAILED_TRANSLATION_CONTAINS_MATCHES = {"(pour le contexte)", "example translation"}


def match_failed_translation_content(text: str) -> bool:
    for match in FAILED_TRANSLATION_CONTAINS_MATCHES:
        if match in text:
            return True
    return False


def is_failed_translation(text: str | None) -> bool:
    return text in FAILED_TRANSLATION_PHRASES or match_failed_translation_content(text)


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


class TranslatedText(OpenAISchema):
    """The result of a translation in the target language."""

    language: Literal["fr"] = Field(
        ..., description="The language of the translation you are providing."
    )
    translated_text: str = Field(..., description="the translated text")


class Result(BaseModel):
    status: Literal["success", "error"]
    result: Word | None = None
    error: dict[str, Any] | None = None


def save_results(results: Iterable[Result], dir: Path, filename: str, suffix: str = ""):
    groups = defaultdict(list)
    for r in results:
        if r.status == "success":
            groups["success"].append(r.result.model_dump())
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


async def translate(mbay: str, english: str, part_of_speech: str | None = None):
    async with sem:
        translation: TranslatedText = call_llm(
            mbay_text=mbay,
            english_text=english,
            part_of_speech=part_of_speech,
        )
    return translation.translated_text


def call_llm(
    mbay_text: str,
    english_text: str,
    part_of_speech: str | None = None,
    verbose=False,
):
    content = render_template(
        TRANSLATION_TMPL,
        mbay_text=mbay_text,
        english_text=english_text,
        part_of_speech=part_of_speech,
    )
    if verbose:
        print(content)
    translation: TranslatedText = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        response_model=TranslatedText,
        messages=[
            {
                "role": "system",
                "content": "You are an expert English to French translator.",
            },
            {"role": "user", "content": content},
        ],
        temperature=0.0,
    )
    if verbose:
        rprint(translation._raw_response)
        rprint(translation)

    return translation


async def fix_record_translation(word: Word, verbose=False) -> Result:
    word = word.model_copy(deep=True)
    try:
        tasks = []
        if is_failed_translation(word.french_translation):
            if verbose:
                rprint(
                    f"Translating {word.headword}, old translation: {word.french_translation}"
                )

            tasks.append(fix_headword(word))

        for example in word.examples:
            if is_failed_translation(example.french):
                if verbose:
                    rprint(
                        f"Translating {example.mbay}, old translation: {example.french}"
                    )

                tasks.append(fix_example_translation(example))

        await asyncio.gather(*tasks)

    except Exception as e:
        return Result(
            status="error",
            error={
                "id": word.id,
                "word": word.model_dump(),
                "error": str(e),
                "type": e.__class__.__name__,
            },
        )

    return Result(status="success", result=word)


async def fix_headword(word: Word) -> None:
    word.french_translation = await translate(
        word.headword,
        word.english_translation,
        word.part_of_speech,
    )


async def fix_example_translation(example: Example) -> None:
    example.french = await translate(example.mbay, example.english)


# TRANSLATION_TMPL = """\
# Translate this entry from a Mbay to English dictionary from English to French:
# mbay text (for context): `{{mbay_text}}`
# {% if part_of_speech %}
# part of speech: `{{part_of_speech}}`
# {% endif %}
# english text: `{{english_text}}`
# """

TRANSLATION_TMPL = """\
Translate this from English to French:
{% if part_of_speech %}
part of speech: `{{part_of_speech}}`
{% endif %}
english text to translate: `{{english_text}}`
"""


T = TypeVar("T")


def batched(iterable: Iterable[T], n: int):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


TMPL_ENV = Environment(trim_blocks=True)


def render_template(template: str, **kwargs: Any) -> str:
    return TMPL_ENV.from_string(template).render(**kwargs)


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
