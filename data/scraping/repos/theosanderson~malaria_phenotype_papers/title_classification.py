from xml.sax.handler import all_properties
from numpy import true_divide
import openai

import pandas
from Bio import Entrez

import json
import tqdm
import sys

Entrez.email = 'theo@theo.io'

handle = open("working/titles.txt", "rt")
all_previous_results = [
    line.split("\t")[0] for line in open("working/titles_classified.txt")
]
all_previous_results = set(all_previous_results)

output = open("working/titles_classified.txt", "a")

for line in tqdm.tqdm(handle):
    try:
        pmid, title = line.strip().split("\t")
    except ValueError:
        continue
    if pmid in all_previous_results:
        print("Already classified", pmid, file=sys.stderr)
        continue
    prompt = title.replace("\n", "").replace("\r", "").replace(
        "\t", "") + "\n\n###\n\n"
    result = openai.Completion.create(
        model="ada:ft-user-kscgj3gd0colhtfqkwmm9sqa-2021-11-12-20-59-38",
        prompt=prompt,
        max_tokens=3,
        temperature=0.0,
        logprobs=3)
    #print(result)
    logprobs = result.choices[0]['logprobs']['top_logprobs']
    tokens = result.choices[0]['logprobs']['tokens']
    main_result = tokens[0].strip()
    main_result_prob = logprobs[0][tokens[0]]

    species_result = tokens[2].strip()
    species_prob = logprobs[2][tokens[2]]

    results = [
        pmid, title, main_result, main_result_prob, species_result,
        species_prob
    ]
    results = [str(x) for x in results]
    print("\t".join(results), file=output)