from dotenv import load_dotenv
load_dotenv(".env")

from database import *
import openai
from openai.error import RateLimitError
import os
import random
import backoff
from typing import List
from peewee import DataError

openai.api_key = os.getenv("OPENAI_APIKEY")

# prepare category getter with exponential backoff baked in
@backoff.on_exception(backoff.expo, RateLimitError)
def get_categories(essay: Essay) -> List[str]:
    summary = essay.summary_en
    if len(summary) > 10_000:
        print(f"[WARNING] Summary truncated to 10k characters")
        summary = essay.summary_en[:10000]

    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an academic library classification engine. When you get a message, you will reply with a comma-separated list of academic domains that best fit the thesis described in the message."},
            {"role": "user", "content": summary }
        ]
    )
    return [c.lower().strip(" ,.!?:;") for c in result["choices"][0]["message"]["content"].split(",")]


for essay_index, essay in enumerate(Essay.select().iterator()):
    if essay.language == "xx" or essay.summary_en is None or len(essay.summary_en) < 50:
        continue

    # skip if already gotten categories with this method
    if EssayCategory.get_or_none(EssayCategory.essay == essay, EssayCategory.method == "openai gpt-3.5-turbo"):
        continue

    # if random.random() > 0.05: # only for a random subset of ~5%
    #     continue

    print(f"[{essay_index + 1}]", essay.title[:120])
    categories = get_categories(essay)

    for index, label in enumerate(categories):
        category, created = Category.get_or_create(name=label)
        link, _ = EssayCategory.get_or_create(
            essay=essay,
            category=category,
            method="openai gpt-3.5-turbo",
        )
        print("  + " if created else "    ", label, sep = "")
