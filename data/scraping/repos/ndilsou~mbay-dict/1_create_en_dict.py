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
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
)
from rich.progress import Progress
from functools import partial
from dotenv import load_dotenv

load_dotenv()
# chat = ChatOpenAI(openai_api_base="http://localhost:9001/v1")
chat = ChatOpenAI()

Record = dict[str, Any]


def main(filename: str):
    filepath = Path(filename)
    with filepath.open("r") as f:
        raw_mbay_content = json.load(f)

    with ThreadPoolExecutor(max_workers=3) as executor:
        run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        start = time.perf_counter()
        results = []
        fn = partial(extract_entry)
        n = 10
        for i, batch in enumerate(batched(raw_mbay_content, n)):
            batch_result = executor.map(fn, batch)
            # batch_result = map(fn, batch)
            results.extend(batch_result)
            rprint(f"{dt.datetime.now()} - batch {i} complete. {n*(i+1)/len(raw_mbay_content)*100:.2f}% complete")
            if i % 10 == 0:
                rprint(f"{dt.datetime.now()} - Checkpointing results to disk (id: {run_id})")
                save_results(results, filepath.parent / "checkpoints", suffix=f"{run_id}_batch_{i}")
        end = time.perf_counter()
        duration = dt.timedelta(seconds=end - start)
        rprint(f"Finished in {duration}")


    save_results(results, filepath.parent) 


def save_results(results: Iterable[Record], dir: Path, suffix: str = ""):
    groups = defaultdict(list)
    for r in results:
        if r["status"] == "success":
            groups["success"].append(r["result"])
        elif r["status"] == "error":
            groups["error"].append(r["error"])
    
    outfilepath = dir / f"mbay_dict_english_clean{suffix}.json"
    with outfilepath.open("w", encoding="utf-8") as f:
        json.dump(groups["success"], f, indent=2, ensure_ascii=False)

    error_filepath = dir / f"mbay_dict_english_clean_errors{suffix}.json"
    with error_filepath.open("w", encoding="utf-8") as f:
        json.dump(groups["error"], f, indent=2, ensure_ascii=False)

def extract_entry(entry: Record, verbose=False, progress: Callable[[], None] | None = None) -> Record:
    _input = prompt.format(
        format_instructions=parser.get_format_instructions(),
        mbay_term=entry["mbay_entry"],
        english_translation=entry["english_entry"],
        raw_html=entry["raw_string"],
    )
    try:
        # print(_input.content)
        # raise ValueError
        output = chat([_input])
        extraction = parser.parse(output.content)
    except Exception as e:
        if progress:
            progress()
        return {
            "status": "error",
            "error": {
                "message": str(e),
                "type": e.__class__.__name__,
                "entry": entry,
                "input": _input.dict(),
            }
        }


    if verbose:
        rprint(extraction)
    result = {
        "id": entry["id"],
        "headword": entry["mbay_entry"],
        "english_translation": extraction.fixed_translation,
        "part_of_speech": extraction.type,
        "sound_file_link": entry["sound_file_link"],
        "page_number": entry["page_number"],
        "examples": [
            {
                "mbay": example.mbay,
                "english": example.english,
                "sound_file_link": example.sound_file,
            }
            for example in extraction.examples
        ],
        "raw_html": entry["raw_string"],
    }
    if progress:
        progress()
    return {
        "status": "success",
        "result": result,
    }

class Example(BaseModel):
    mbay: str = Field(..., description="The mbay text")
    english: str = Field(..., description="The english translation")
    sound_file: str | None = Field(
        None, description="The optional sound file associated with the example"
    )


class Extraction(BaseModel):
    examples: list[Example] = Field(
        ..., description="A list of examples associated with the dictionary entry. There is often multiple examples for each entry."
    )
    fixed_translation: str = Field(
        ..., description="The fixed translation with any mangled suffix corrected."
    )
    type: str | None = Field(
        None, description="The type of the dictionary entry, if recorded."
    )


# parser =
parser = OutputFixingParser.from_llm(
    parser=PydanticOutputParser(pydantic_object=Extraction), llm=ChatOpenAI()
)
EXTRACTION_TMPL_LLAMA2 = """\
<<SYS> {{format_instructions}} <</SYS> You are given a sample from a Mbay to English dictionary. The main term of the entry has been extracted. additional examples are present in the raw string given to you. Extract all the Mbay to English example pairs. Notice that there is often multiple examples for each entry.

Mbay Term: "ngōn-kó̰o̰"
English Translation: "sibling, brother or sister."
Raw HTML:
"<div class=\"def\">\n<span class=\"word\">\n<span class=\"w sara-bagirmi-lang\">\n<a class=\"play-sound\" onclick=\"playSound('Sound/NgonKooNW.mp3')\"> ngōn-kó̰o̰</a>\n</span>\n[ŋgō̰nkó̰o̰]\n</span>\n<span class=\"type\">NT  </span>\nsibling, brother or sister.\n<span class=\"sara-bagirmi-lang\">\n<a class=\"play-sound\" onclick=\"playSound('Sound/NgonKooNSS1.mp3')\"> Ngōn-kó̰o̰-í à dān-ī súl tɨ́ nà ngōn-bɔ̀ɔ̄-í à là gìdɨ̀ bɔ̀r.</a> </span>\n{Proverb}  Your brother will accompany you to prison, but your half-brother will amble about the walls.\n</div>"
Result:
{
"examples": [
{"mbay": "Ngōn-kó̰o̰-í à dān-ī súl tɨ́ nà ngōn-bɔ̀ɔ̄-í à là gìdɨ̀ bɔ̀r.", "english": "{Proverb}  Your brother will accompany you to prison, but your half-brother will amble about the walls.", "sound_file": "Sound/NgonKooNSS1.mp3"}
],
"fixed_translation": "sibling, brother or sister.",
"type": "NT"
}

Mbay Term: "à"
English Translation: "particle marking an interrogative sentence (alternate to"
Raw HTML:
"<div class=\"def\">\n<span class=\"word\">\n<span class=\"w sara-bagirmi-lang\">\n<a class=\"play-sound\" onclick=\"playSound('Sound/MBAY01A11.mp3')\"> à</a>\n</span>\n(Syn: è)\n</span>\n<span class=\"type\">INT  </span>\nparticle marking an interrogative sentence (alternate to\n<span class=\"sara-bagirmi-lang\">wà</span>).\n<span class=\"sara-bagirmi-lang\">\n<a class=\"play-sound\" onclick=\"playSound('Sound/MBAY01A26.mp3')\"> Bà̰ý à?</a> </span>\nHow goes it?\n<span class=\"sara-bagirmi-lang\">\n<a class=\"play-sound\" onclick=\"playSound('Sound/MBAY01A27.mp3')\"> ī-tò kàrī à?</a> </span>\nHow are you?\n</div>"
Result:
{
"examples": [
{"mbay": "Bà̰ý à?", "english": "How goes it?", "sound_file": "Sound/MBAY01A26.mp3"},
{"mbay": "ī-tò kàrī à?", "english": "How are you?", "sound_file": "Sound/MBAY01A27.mp3"}
],
"fixed_translation": "particle marking an interrogative sentence (alternate to wà)",
"type": "INT"
}

Mbay Term: "ngōn-kɨ̀lē"
English Translation: "key (see also"
Raw HTML:
"<div class=\"def\">\n<span class=\"word\">\n<span class=\"w sara-bagirmi-lang\">\n ngōn-kɨ̀lē\n</span>\n[ŋgō̰nkɨ̀lē]\n(Syn: =kɨ̀léè, làkér)\n</span>\n<span class=\"type\">NI  </span>\nkey (see also\n<span class=\"sara-bagirmi-lang\">kó̰o̰-kɨ̀lē</span> 'lock').\n<span class=\"sara-bagirmi-lang\">\n ādɨ̄-m̄ ngōn-kɨ̀lē ādɨ̄ m̄-tèē-ň tà-ɗóbɨ́. </span>\nGive me the key so I can open the door.\n</div>"
Result:
{
"examples": [
{"mbay": "ādɨ̄-m̄ ngōn-kɨ̀lē ādɨ̄ m̄-tèē-ň tà-ɗóbɨ́.", "english": "Give me the key so I can open the door.", "sound_file": null}
],
"fixed_translation": "key (see also kó̰o̰-kɨ̀lē 'lock').",
"type": "NI"
}

Mbay Term: {{mbay_term}}
English Translation: {{english_translation}}
Raw HTML:
{{raw_html}}

Result:"""

EXTRACTION_TMPL = """\
You are given a sample from a Mbay to English dictionary. The main term of the entry has been extracted. additional examples are present in the raw string given to you. Extract all the Mbay to English example pairs.

{{format_instructions}}

Mbay Term: "ngōn-kó̰o̰"
English Translation: "sibling, brother or sister."
Raw HTML:
"<div class=\"def\">\n<span class=\"word\">\n<span class=\"w sara-bagirmi-lang\">\n<a class=\"play-sound\" onclick=\"playSound('Sound/NgonKooNW.mp3')\"> ngōn-kó̰o̰</a>\n</span>\n[ŋgō̰nkó̰o̰]\n</span>\n<span class=\"type\">NT  </span>\nsibling, brother or sister.\n<span class=\"sara-bagirmi-lang\">\n<a class=\"play-sound\" onclick=\"playSound('Sound/NgonKooNSS1.mp3')\"> Ngōn-kó̰o̰-í à dān-ī súl tɨ́ nà ngōn-bɔ̀ɔ̄-í à là gìdɨ̀ bɔ̀r.</a> </span>\n{Proverb}  Your brother will accompany you to prison, but your half-brother will amble about the walls.\n</div>"
Result:
{
"examples": [
{"mbay": "Ngōn-kó̰o̰-í à dān-ī súl tɨ́ nà ngōn-bɔ̀ɔ̄-í à là gìdɨ̀ bɔ̀r.", "english": "{Proverb}  Your brother will accompany you to prison, but your half-brother will amble about the walls.", "sound_file": "Sound/NgonKooNSS1.mp3"}
],
"fixed_translation": "sibling, brother or sister.",
"type": "NT"
}

Mbay Term: "ngōn-kɨ̀lē"
English Translation: "key (see also"
Raw HTML:
"<div class=\"def\">\n<span class=\"word\">\n<span class=\"w sara-bagirmi-lang\">\n ngōn-kɨ̀lē\n</span>\n[ŋgō̰nkɨ̀lē]\n(Syn: =kɨ̀léè, làkér)\n</span>\n<span class=\"type\">NI  </span>\nkey (see also\n<span class=\"sara-bagirmi-lang\">kó̰o̰-kɨ̀lē</span> 'lock').\n<span class=\"sara-bagirmi-lang\">\n ādɨ̄-m̄ ngōn-kɨ̀lē ādɨ̄ m̄-tèē-ň tà-ɗóbɨ́. </span>\nGive me the key so I can open the door.\n</div>"
Result:
{
"examples": [
{"mbay": "ādɨ̄-m̄ ngōn-kɨ̀lē ādɨ̄ m̄-tèē-ň tà-ɗóbɨ́.", "english": "Give me the key so I can open the door.", "sound_file": null}
],
"fixed_translation": "key (see also kó̰o̰-kɨ̀lē 'lock').",
"type": "NI"
}

Mbay Term: {{mbay_term}}
English Translation: {{english_translation}}
Raw HTML:
{{raw_html}}

Result:"""


prompt = HumanMessagePromptTemplate.from_template(
    template=EXTRACTION_TMPL_LLAMA2,
    input_variables=["mbay_term", "english_translation", "raw_html"],
    template_format="jinja2",
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

T  = TypeVar("T")

def batched(iterable: Iterable[T], n: int):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

if __name__ == "__main__":
    main(sys.argv[1])
