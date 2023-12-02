import ndjson
import json
import sys
import os

from tqdm import tqdm
import numpy as np
import openai

from transformers import AutoTokenizer

IN_DIR = "../docgen_parse/docgen_export_with_formal_statement.jsonl"

tokenizer = AutoTokenizer.from_pretrained("gpt2")

with open(IN_DIR) as f:
    data = ndjson.load(f)

total = 0
for x in tqdm(data):
    text = (
        "/-- " + x["doc_string"] + " -/" + "\n" + x["formal_statement"]
        if x["doc_string"]
        else x["formal_statement"]
    )

    count = len(tokenizer(text)['input_ids'])

    total += count

print(total)


