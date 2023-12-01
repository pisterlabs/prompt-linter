# 03 Augment LLM
# Input: `data/02_backtranslate_english.csv`
# Output: `data/03_augment_llm_gec.csv` `data/03_augment_llm_paraphrase.csv`

import os
import threading

import nltk

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

import pandas as pd
import tqdm.auto as tqdm
from rich import print
from llama_cpp import Llama

# Load api key
api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = api_key

nltk.download("punkt", quiet=True)

BATCH = 1

# Load data
df = pd.read_csv("data/03_augment_llm_input.csv") # .sample(frac=1).reset_index(drop=True)

# AI Constants
# These original messages are generated from the same model, however we include them inside the chat to make sure the model returns in the correct format.
#model = "/home/parin/ChatModels/platypus2-70b-instruct.ggmlv3.q2_K.bin"
model = "/home/parin/ChatModels/openorca-platypus2-13b.ggmlv3.q2_K.bin"

example_grammar_error_correction_message = """### Instruction:
You are a highly skilled language model AI that returns only one line of grammatically perfect text. Your task is to evaluate the text below and correct its grammar. Even if the text is incomplete or unintelligible, YOU MUST make a grammatical correction, you can make assumptions about the intended meaning. If the text is grammatically correct, do not change it. Your output should be presented WITH ONLY the corrected text IN ONE LINE and without any extra dialogue from you. Do not use any new lines in your output. Your output should only have one line.

### Input:
Phatthira Sarutpong Phokin What is the date of birth? Phatthira Sarutpong Phokin What are you playing? What is the career of Phatthira Teerathiyapong Phokin? Phatthira Sarutpong Phokin Graduated from what country? Father Cleopatra who? P?

### Response:
What is Phatthira Sarutpong Phokin's date of birth? What are you playing, Phatthira Sarutpong Phokin? What is Phatthira Teerathiyapong Phokin's career? Phatthira Sarutpong Phokin graduated from what country? Who is Cleopatra's father? P?

### Input:
Emperor, Bennesanus Galseus or Gaisus Viyas, Bonianus Gallus, full name: Gaius vibonianus gallus (206 - August 1990) Bennenus Galsen is the emperor of the Roman Empire that reigned in 1918 with the Emperor Homius and between 1918 to August. 1990, in collaboration with the son of Emperor Voluzanus?

### Response:
Emperor Bennesanus Galseus or Gaisus Viyas, Bonianus Gallus, full name: Gaius Vibonianus Gallus (206 - August 1990). Bennenus Galseus is the emperor or of the Roman Empire that reigned in 1918 with Emperor Homius in 1918 and the son of Emperor Voluzanus Voluzanus in August 1990.

### Input:
"""

example_paraphrase_message = """### Instruction:
You are a highly skilled language model AI that returns only one line of linguistically diverse paraphrased text. Your task is to perform two specific actions on a given text. First, evaluate each text and make sure it's grammatically correct. If a text is not grammatically correct, fix it. Then, ALWAYS paraphrase the text while maintaining its original meaning. Your output should be presented WITH ONLY the paraphrased text IN ONE SINGLE LINE, without any extra dialouge from you. Do not use any new lines in your output. Only write in one line.

### Input:
Phatthira Sarutpong Phokin What is the date of birth? Phatthira Sarutpong Phokin What are you playing? What is the career of Phatthira Teerathiyapong Phokin? Phatthira Sarutpong Phokin Graduated from what country? Who is Cleopatra's father? A? P.

### Response:
Phatthira Sarutpong Phokin, what is your date of birth? Phokin, What are you playing? What is Phatthira's career and from which country did Phokin graduate? Lastly, Who is the father of Cleopatra? A? P.

### Input:
Emperor, Bennesanus Galseus or Gaisus Viyas, Bonianus Gallus, full name: Gaius vibonianus gallus (206 - August 1990) Bennenus Galsen is the emperor of the Roman Empire that reigned in 1918 with the Emperor Homius and between 1918 to August. 1990, in collaboration with the son of Emperor Voluzanus?

### Response:
Emperor Bennesanus Galseus, also known as Gaisus Viyas and Bonianus Gallus, had a full name of Gaius Vibonianus Gallus and reigned from 206 to August 1990. He was the Roman Emperor who collaborated with Emperor Homius in 1918 and with the latter's son Voluzanus between 1918 and August 1990.

### Input:
"""

get_gec_msg = lambda x: example_grammar_error_correction_message + x + "\n\n### Response:\n"
get_paraphrase_msg = lambda x: example_paraphrase_message + x + "\n\n### Response:\n"

model = Llama(model, seed=42, n_ctx=4096, n_gpu_layers=60, n_threads=1, use_mlock=True, last_n_tokens_size=128)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = int(len(string) * 0.6)
    return num_tokens

def break_into_chunks(text: str, max_tokens: int = 1024, encoding_name: str = "gpt-3.5-turbo") -> list:
    """Breaks the text into chunks that fit within the model's context."""
    chunks = []
    current_chunk = ""
    words = text.split()
    
    for word in words:
        temp_chunk = current_chunk + " " + word if current_chunk else word
        token_count = num_tokens_from_string(temp_chunk, encoding_name)
        
        if token_count <= max_tokens:
            current_chunk = temp_chunk
        else:
            chunks.append(current_chunk)
            current_chunk = word
    
    # Adding the last chunk if there's anything left
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Grammar Error Correction
def grammar_error_correction():
    # Check if output exists
    if os.path.exists("data/03_augment_llm_gec.csv"):
        data = pd.read_csv("data/03_augment_llm_gec.csv").to_dict("records")
        completed_ids = set([x["context"] for x in data])
    else:
        data = []
        completed_ids = set()

    # Loop through each row in the dataframe using batches of 15
    for i in tqdm.tqdm(range(0, len(df), BATCH), desc="Correcting grammar"):
        # Get the next 25 rows
        batch = df.iloc[i : i + BATCH].copy()
        batch = batch[~batch["context"].isin(completed_ids)]

        if len(batch) == 0:
            continue

        # Shuffle the rows
        batch = batch.sample(frac=1).reset_index(drop=True)
        to_correct = "\n".join(batch["en_aug"].tolist())
        to_correct_chunks = break_into_chunks(to_correct)
        final = ""
        for chunk in to_correct_chunks:
            attempt = 0
            while True:
                try:
                    response = model(get_gec_msg(chunk), stop="\n", max_tokens=0)
                    raw_corrected = response["choices"][0]["text"]
                    corrected = raw_corrected.strip().split("\n")
                    corrected = [x.strip() for x in corrected if x.strip() != ""]
                    assert len(corrected) == len(batch)
                    # Assume batch is one
                    final += " " + corrected[0].strip()
                    break
                except AssertionError:
                    print(f"Model Returned:\n{raw_corrected}")
                    print(f"Error:\n{batch['en_aug'].to_list()}")
                    print(f"Attempt {attempt+1} of 6")
                    if attempt == 6:
                        raise
                    attempt += 1

        # Add to data
        for idx, corrected_sentence in enumerate([final]):
            data.append(
                {
                    "context": batch.iloc[idx]["context"],
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
        completed_ids = set([x["context"] for x in data])
    else:
        data = []
        completed_ids = set()

    # Loop through each row in the dataframe using batches of 15 
    for i in tqdm.tqdm(range(0, len(df), BATCH), desc="Paraphrasing questions"):
        batch = df.iloc[i : i + BATCH]
        batch = batch[~batch["context"].isin(completed_ids)]
        if len(batch) == 0:
            continue

        # Shuffle the rows
        batch = batch.sample(frac=1).reset_index(drop=True)
        to_paraphrase = "\n".join(batch["en_aug"].tolist())
        to_paraphrase_chunks = break_into_chunks(to_paraphrase)
        final = ""
        for chunk in to_paraphrase_chunks:
            attempt = 0
            while True:
                try:
                    response = model(get_paraphrase_msg(chunk), stop="\n", max_tokens=0)
                    raw_paraphrases = response["choices"][0]["text"]
                    paraphrases = raw_paraphrases.strip().split("\n")
                    paraphrases = [x.strip() for x in paraphrases if x.strip() != ""]
                    assert len(paraphrases) == len(batch)
                    # Assume batch is one
                    final += " " + paraphrases[0].strip()
                    break
                except AssertionError:
                    print(f"Model Returned:\n{raw_paraphrases}")
                    print(f"Error:\n{batch['en_aug'].to_list()}")
                    print(f"Attempt {attempt+1} of 6")
                    if attempt == 6:
                        raise
                    attempt += 1

        # Add to data
        for idx, paraphrased in enumerate([final]):
            data.append(
                {"context": batch.iloc[idx]["context"], "en_llm_paraphrase_aug": paraphrased}
            )
        pd.DataFrame(data).to_csv("data/03_augment_llm_paraphrase.csv", index=False)


if __name__ == "__main__":
#    thread_1 = threading.Thread(target=grammar_error_correction)
#    thread_2 = threading.Thread(target=paraphrase)

    # Start both threads
#    thread_1.start()
#    thread_2.start()

    # Wait for both threads to finish
#    thread_1.join()
#    thread_2.join()
    grammar_error_correction()
    paraphrase()

    # Merge the data from both threads
    gec = pd.read_csv("data/03_augment_llm_gec.csv")
    paraphrased = pd.read_csv("data/03_augment_llm_paraphrase.csv")
    merge = pd.merge(gec, paraphrased, on="context")
    final = pd.merge(df, merge, on="context")
    
    # Sanity check that rows were not lost
    assert len(gec) == len(paraphrased)
    assert len(gec) == len(merge)
    assert len(gec) == len(final)
    assert len(final) == len(df)

    # Save the data
    final.to_csv("data/03_augment_llm.csv", index=False)
