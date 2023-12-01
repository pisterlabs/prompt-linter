import pandas as pd
import numpy as np
from Bio import Entrez
import openai
import sys
import tqdm
import gzip
# initialize some default parameters
Entrez.email = 'theo@theo.io'  # provide your email address
done = set()
if True:
    handle = open("working/abstracts_classified.txt", "rt")
    
    for line in tqdm.tqdm(handle):
        items = line.strip().split("\t")
        pmid = items[0]

        done.add(pmid)


def do_classification(pmid):

    try:
        handle = Entrez.efetch(db='pubmed',
                               id=pmid,
                               retmode="xml",
                               rettype="abstract")
        record = Entrez.read(handle)
        abstract = record['PubmedArticle'][0]['MedlineCitation']['Article'][
            'Abstract']['AbstractText']
    except Exception:
        return None
    abstract = str(abstract)

    prompt = abstract + "\n\n###\n\n"
    result = openai.Completion.create(
        model="ada:ft-user-kscgj3gd0colhtfqkwmm9sqa-2021-11-12-17-03-42",
        prompt=prompt,
        max_tokens=3,
        logprobs=3,
        temperature=0.0)
    logprobs = result.choices[0]['logprobs']['top_logprobs']
    tokens = result.choices[0]['logprobs']['tokens']
    main_result = tokens[0].strip()
    main_result_prob = logprobs[0][tokens[0]]

    species_result = tokens[2].strip()
    species_prob = logprobs[2][tokens[2]]

    #print(abstract)

    return main_result, main_result_prob, species_result, species_prob


columns = [
    'pmid',
    'title',
    'main_result',
    'main_result_prob',
    'secondary_result',
    'secondary_result_prob',
]

by_title = pd.read_csv("./working/titles_classified.txt",
                       names=columns,
                       sep="\t")
by_title['raw_prob'] = np.exp(by_title['main_result_prob'])

other = by_title[by_title.main_result == "other"]
phenotype = by_title[by_title.main_result == "phenotype"]

uncertain_pos = other.raw_prob < 0.95

uncertain = other[uncertain_pos]

to_consider = phenotype['pmid'].to_list() + uncertain['pmid'].to_list()
# dedupe
to_consider = list(set(to_consider))

out_handle = open("working/abstracts_classified.txt", "a")

for pmid in tqdm.tqdm(to_consider):
    if str(pmid) in done:
        continue
    result = do_classification(pmid)
    if result is None:
        #log to stderr
        print("{} not found".format(pmid), file=sys.stderr)

        continue
    output = [pmid] + list(result)
    print("\t".join([str(x) for x in output]), file=out_handle)
