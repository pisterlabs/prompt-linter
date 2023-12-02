# 03 Augment LLM
# Input: `data/02_backtranslate_english.csv`
# Output: `data/03_augment_llm_gec.csv` `data/03_augment_llm_paraphrase.csv`

import os
import threading

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

import pandas as pd
import tqdm.auto as tqdm
from rich import print

# Load api key
api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = api_key

# Load data
df = pd.read_csv("data/02_backtranslate_english.csv")

# AI Constants
# These original messages are generated from the same model, however we include them inside the chat to make sure the model returns in the correct format.
model = "gpt-3.5-turbo-0301"
example_grammar_error_correction_messages = [
    {
        "role": "system",
        "content": "You are a highly skilled language model AI. Your task is to evaluate EACH AND EVERY question from a given list and correct its grammar. Even if a question is incomplete or unintelligible, YOU MUST make a grammatical correction, you can make assumptions about the intended meaning. If the question is grammatically correct, do not change it. In the case of duplicate questions, return them IN THE SAME ORDER as presented with no removal. Your output should be presented WITH EACH AND EVERY question with ONLY each question written on a new line.",
    },
    {
        "role": "user",
        "content": "Phatthira Sarutpong Phokin What is the date of birth?\nPhatthira Sarutpong Phokin What are you playing?\nWhat is the career of Phatthira Teerathiyapong Phokin?\nPhatthira Sarutpong Phokin Graduated from what country?\nWho is Cleopatra's father?",
    },
    {
        "role": "assistant",
        "content": "What is Phatthira Sarutpong Phokin's date of birth?\nWhat are you playing, Phatthira Sarutpong Phokin?\nWhat is Phatthira Teerathiyapong Phokin's career?\nPhatthira Sarutpong Phokin graduated from what country?\nWho is Cleopatra's father?",
    },
]

example_paraphrase_messages = [
    {
        "role": "system",
        "content": "You are a highly skilled language model AI. Your task is to perform two specific actions on a given list of questions. First, evaluate each question and make sure it's grammatically correct. If a question is not grammatically correct, fix it. Then, ALWAYS paraphrase each question while maintaining its original meaning.  In the case of duplicate questions, return them IN THE SAME ORDER as presented with no removal. Your output should be presented WITH EACH AND EVERY question with ONLY each paraphrased question written on a new line.",
    },
    {
        "role": "user",
        "content": "Phatthira Sarutpong Phokin What is the date of birth?\nPhatthira Sarutpong Phokin What are you playing?\nWhat is the career of Phatthira Teerathiyapong Phokin?\nPhatthira Sarutpong Phokin Graduated from what country?\nWho is Cleopatra's father?",
    },
    {
        "role": "assistant",
        "content": "Phatthira Sarutpong Phokin, what is your date of birth?\nWhat are you playing, Phatthira Sarutpong Phokin?\nWhat is Phatthira Teerathiyapong Phokin's career?\nFrom which country did Phatthira Sarutpong Phokin graduate?\nWho is the father of Cleopatra?",
    },
]


# Add exponential backoff to completion
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


# Grammar Error Correction
def grammar_error_correction():
    # Check if output exists
    if os.path.exists("data/03_augment_llm_gec.csv"):
        data = pd.read_csv("data/03_augment_llm_gec.csv").to_dict("records")
        completed_ids = set([x["id"] for x in data])
    else:
        data = []
        completed_ids = set()

    # Loop through each row in the dataframe using batches of 15
    for i in tqdm.tqdm(range(0, len(df), 15), desc="Correcting grammar"):
        # Get the next 25 rows
        batch = df.iloc[i : i + 15].copy()
        batch = batch[~batch["id"].isin(completed_ids)]

        if len(batch) == 0:
            continue

        # Shuffle the rows
        attempt = 0
        while True:
            batch = batch.sample(frac=1).reset_index(drop=True)
            to_correct = "\n".join(batch["en_aug"].tolist())
            try:
                response = completion_with_backoff(
                    model=model,
                    messages=[
                        *example_grammar_error_correction_messages,
                        {"role": "user", "content": to_correct},
                    ],
                )
                raw_corrected = response["choices"][0]["message"]["content"]
                corrected = raw_corrected.split("\n")
                corrected = [x.strip() for x in corrected if x.strip() != ""]
                assert len(corrected) == len(batch)
                break
            except AssertionError:
                print(f"Model Returned:\n{raw_corrected}")
                print(f"Error:\n{batch['en_aug'].to_list()}")
                print(f"Attempt {attempt+1} of 6")
                if attempt == 6:
                    raise
                attempt += 1

        # Add to data
        for idx, corrected_sentence in enumerate(corrected):
            data.append(
                {
                    "id": batch.iloc[idx]["id"],
                    "en_llm_gec_aug": corrected_sentence,
                }
            )

        # Save data
        pd.DataFrame(data).to_csv("data/03_augment_llm_gec.csv", index=False)


# Paraphrase
def paraphrase():
    # Check if output exists
    if os.path.exists("data/03_augment_llm_paraphrase.csv"):
        data = pd.read_csv("data/03_augment_llm_paraphrase.csv").to_dict("records")
        completed_ids = set([x["id"] for x in data])
    else:
        data = []
        completed_ids = set()

    # Loop through each row in the dataframe using batches of 15 
    for i in tqdm.tqdm(range(0, len(df), 15), desc="Paraphrasing questions"):
        batch = df.iloc[i : i + 15]
        batch = batch[~batch["id"].isin(completed_ids)]
        if len(batch) == 0:
            continue

        # Shuffle the rows
        attempt = 0
        while True:
            batch = batch.sample(frac=1).reset_index(drop=True)
            to_paraphrase = "\n".join(batch["en_aug"].tolist())
            try:
                response = completion_with_backoff(
                    model=model,
                    messages=[
                        *example_paraphrase_messages,
                        {"role": "user", "content": to_paraphrase},
                    ],
                )
                raw_paraphrases = response["choices"][0]["message"]["content"]
                paraphrases = raw_paraphrases.split("\n")
                paraphrases = [x.strip() for x in paraphrases if x.strip() != ""]
                assert len(paraphrases) == len(batch)
                break
            except AssertionError:
                print(f"Model Returned:\n{raw_paraphrases}")
                print(f"Error:\n{batch['en_aug'].to_list()}")
                print(f"Attempt {attempt+1} of 6")
                if attempt == 6:
                    raise
                attempt += 1

        # Add to data
        for idx, paraphrased in enumerate(paraphrases):
            data.append(
                {"id": batch.iloc[idx]["id"], "en_llm_paraphrase_aug": paraphrased}
            )
        pd.DataFrame(data).to_csv("data/03_augment_llm_paraphrase.csv", index=False)


if __name__ == "__main__":
    thread_1 = threading.Thread(target=grammar_error_correction)
    thread_2 = threading.Thread(target=paraphrase)

    # Start both threads
    thread_1.start()
    thread_2.start()

    # Wait for both threads to finish
    thread_1.join()
    thread_2.join()

    # Merge the data from both threads
    gec = pd.read_csv("data/03_augment_llm_gec.csv")
    paraphrase = pd.read_csv("data/03_augment_llm_paraphrase.csv")
    merge = pd.merge(gec, paraphrase, on="id")
    final = pd.merge(df, merge, on="id")
    
    # Sanity check that rows were not lost
    assert len(gec) == len(paraphrase)
    assert len(gec) == len(merge)
    assert len(gec) == len(final)
    assert len(final) == len(df)

    # Save the data
    final.to_csv("data/03_augment_llm.csv", index=False)
