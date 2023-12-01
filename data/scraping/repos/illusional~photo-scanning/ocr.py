import json
import os
import re
from glob import glob

import openai
from ocrmac import ocrmac

end_filename_regex = re.compile(r"(_[a|b])?\.jpg$")

openai.organization = os.getenv("OPENAI_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")
PRICE_PER_REQUEST_TOKEN = 0.0015 / 1000
PRICE_PER_OUTPUT_TOKEN = 0.002 / 1000
total_cost = 0


class MetKeys:
    annotation = "annotations"
    text_raw = "text-raw"
    text_corrected = "text-corrected"


def get_meta_file_path(photo_filename: str):
    """
    Test filename matching

    >>> get_meta_file_path('test.jpg')
    'test.meta.json'

    >>> get_meta_file_path('test_a.jpg')
    'test.meta.json'

    >>> get_meta_file_path('test_b.jpg')
    'test.meta.json'
    """
    return (
        photo_filename.removesuffix(".jpg").removesuffix("_a").removesuffix("_b")
        + ".meta.json"
    )


def prepare_all_text_for_ocr(directory, force: bool = False):
    for photo in glob(directory + "*_b.jpg"):
        ocr_photo(photo, force=force)


def ocr_photo(photo, force: bool = False) -> dict:
    meta_path = photo.replace(".jpg", ".meta.json")

    if not os.path.exists(photo):
        return None

    if not force and os.path.exists(meta_path):
        return

    print(f"Processing {photo}")

    annotations = ocrmac.OCR(photo).recognize()

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            d = json.load(f)
    else:
        d = {}

    d[MetKeys.annotation] = annotations
    d[MetKeys.text_raw] = " ".join(a[0] for a in annotations)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


# region chatgpt


def report_tokens_used(prompt_tokens, completion_tokens):
    global total_cost
    total_cost += prompt_tokens * PRICE_PER_REQUEST_TOKEN
    total_cost += completion_tokens * PRICE_PER_OUTPUT_TOKEN
    print(f"Total cost so far: ${total_cost:.2f}")


def prompt_generator(text):
    return f"""
I OCR'd the back of an old photo, and the result was: "{text}"
Can you reformat this message to fix any errors you think occurred, and if you're confident you found a date, can you display it on the last line with the format: "Date: yyyy-mm-dd HH:MM". If you don't know, say "Date: N/A".
Please just return the fixed statement, and don't return any commentary.""".strip()


def fix_text_from_chatgpt(meta_path, force: bool = False):
    """Fixes text from chatgpt, and returns the fixed text."""

    if not os.path.exists(meta_path):
        print(f"Could not process {meta_path} as it did not exist")
        # skip if we've already done this
        return
    print(f"Fixing text from chatgpt {meta_path.replace('.meta.json', '')}")
    with open(meta_path, encoding="utf-8") as f:
        d = json.load(f)

    if MetKeys.text_corrected in d and not force:
        return

    text_raw = d.get(MetKeys.text_raw)
    if not text_raw:
        print(f"Metadata exists, but no text raw for {meta_path}")
        return

    prompt = prompt_generator(text_raw)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    chat_gpt_text = response["choices"][0]["message"]["content"].strip('"')

    d[MetKeys.text_corrected] = chat_gpt_text
    with open(meta_path, "w+") as f:
        json.dump(d, f)

    report_tokens_used(
        response["usage"]["prompt_tokens"], response["usage"]["completion_tokens"]
    )


def fix_all_ocrd_text(directory):
    files = glob(directory + "*.meta.json")
    print(f"Fixing {len(files)} messages")
    for meta_path in files:
        with open(meta_path) as f:
            fix_text_from_chatgpt(meta_path)


# endregion chatgpt


if __name__ == "__main__":
    import sys

    prepare_all_text_for_ocr(sys.argv[1])
    # fix_all_ocrd_text(new_dir)
