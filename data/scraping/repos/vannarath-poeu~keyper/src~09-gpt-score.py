import openai
import dotenv
import os
import argparse
import pathlib
import json
import re
import time
import copy

from utils.evaluation import evaluate
from utils.nlp import stem_keywords

def score():
    test_jsonl = f"data/midas/ldkp3k/test.jsonl"
    data_path = "output/gpt/processed"
    processed_file = f"{data_path}/keyphrases.json"

    assert os.path.exists(test_jsonl), f"File {test_jsonl} does not exist"
    assert os.path.exists(processed_file), f"File {processed_file} does not exist"
    
    with open(test_jsonl, "r") as f:
        test = f.readlines()
    
    with open(processed_file, "r") as f:
        keyphrase_dict = json.load(f)
    
    results = {
        k : {
            "precision@5": 0,
            "recall@5": 0,
            "fscore@5": 0,
            "precision@10": 0,
            "recall@10": 0,
            "fscore@10": 0,
        }
        for k in ["abstractive", "extractive", "combined"]
    }
    
    num_records = len(keyphrase_dict)
    print(f"Number of records: {num_records}")

    processed = 0
    
    for k, phrases in keyphrase_dict.items():
        i = int(k)

        test[i] = json.loads(test[i])
        abstractive_keyphrases = test[i]["abstractive_keyphrases"]
        extractive_keyphrases = test[i]["extractive_keyphrases"]

        abstractive_keyphrases = stem_keywords(abstractive_keyphrases)
        extractive_keyphrases = stem_keywords(extractive_keyphrases)
        combined_keyphrases = abstractive_keyphrases + extractive_keyphrases

        predicted_keyphrases = stem_keywords(phrases)

        for k in [5, 10]:
            p, r, f = evaluate(predicted_keyphrases[:k], abstractive_keyphrases)
            results["abstractive"][f"precision@{k}"] += p
            results["abstractive"][f"recall@{k}"] += r
            results["abstractive"][f"fscore@{k}"] += f

        for k in [5, 10]:
            p, r, f = evaluate(predicted_keyphrases[:k], extractive_keyphrases)
            results["extractive"][f"precision@{k}"] += p
            results["extractive"][f"recall@{k}"] += r
            results["extractive"][f"fscore@{k}"] += f

        for k in [5, 10]:
            p, r, f = evaluate(predicted_keyphrases[:k], combined_keyphrases)
            results["combined"][f"precision@{k}"] += p
            results["combined"][f"recall@{k}"] += r
            results["combined"][f"fscore@{k}"] += f
        
        processed += 1
        print(f"Processed {processed} documents", end="\r")

        temp = copy.deepcopy(results)

        for k in temp.keys():
            for score in temp[k].keys():
                temp[k][score] /= (i+1)
        temp["num_docs"] = processed
        json.dump(temp, open(f"{data_path}/scores-n={num_records}.json", "w"), indent=4)
    
    


if __name__ == "__main__":
    # Example: python3 src/09-gpt-score.py
    score()