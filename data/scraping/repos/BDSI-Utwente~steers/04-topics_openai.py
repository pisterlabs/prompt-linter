from dotenv import load_dotenv
load_dotenv(".env")

from database import *
import openai
from openai.error import RateLimitError
import os
import random
import backoff
from peewee import DataError

openai.api_key = os.getenv("OPENAI_APIKEY")

# prepare topic getter with exponential backoff baked in
@backoff.on_exception(backoff.expo, RateLimitError)
def get_topics(essay: Essay):
    summary = essay.summary_en
    if len(summary) > 10_000:
        print(f"[WARNING] Summary truncated to 10k characters")
        summary = essay.summary_en[:10000]

    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a topic extraction engine. When you get a message, you will reply with a comma-separated list of up to 8 topics and concepts that are most relevant to that message."},
            {"role": "user", "content": summary }
        ]
    )
    return [c.lower().strip(" ,.!?:;") for c in result["choices"][0]["message"]["content"].split(",")]


for essay_index, essay in enumerate(Essay.select().iterator()):
    if essay.language == "xx" or essay.summary_en is None or len(essay.summary_en) < 50:
        continue

    # skip if already gotten topics with this method
    if EssayTopic.get_or_none(EssayTopic.essay == essay, EssayTopic.method == "openai gpt-3.5-turbo"):
        continue

    # if random.random() > 0.05: # only for a random subset of ~5%
    #     continue

    print(f"[{essay_index + 1}]", essay.title[:120])

    labels = get_topics(essay)
    attempt = 1
    while any(map(lambda label: len(label) > 50, labels)): 
        if attempt < 5:
            print(f"[WARNING] unusually long topic(s), retrying...")
            attempt += 1
        else:
            print(f"[ERROR] unusually long topic(s) after {attempt} retries, giving up!")
            for label in labels:
                print(f"\t{label}")
        labels = get_topics(essay)

    for index, label in enumerate(labels):
        topic, created = Topic.get_or_create(name=label)
        link, _ = EssayTopic.get_or_create(
            essay=essay,
            topic=topic,
            method="openai gpt-3.5-turbo",
        )
        print("  + " if created else "    ", label, sep = "")
