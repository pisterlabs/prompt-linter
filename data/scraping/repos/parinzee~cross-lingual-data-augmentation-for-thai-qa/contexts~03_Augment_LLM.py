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
import tiktoken

# Load api key
api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = api_key

nltk.download("punkt", quiet=True)

BATCH = 1

# Load data
df = pd.read_csv("data/03_augment_llm_input.csv") # .sample(frac=1).reset_index(drop=True)

# AI Constants
# These original messages are generated from the same model, however we include them inside the chat to make sure the model returns in the correct format.
model = "gpt-3.5-turbo-16k-0613"
# example_grammar_error_correction_messages = [
#     {
#         "role": "system",
#         "content": "You are a highly skilled language model AI. Your task is to evaluate the LIST of text below (separated by newlines) and correct its grammar. Even if the text is incomplete or unintelligible, YOU MUST make a grammatical correction, you can make assumptions about the intended meaning. If the text is grammatically correct, do not change it. Your output should be presented with ONLY the corrected LIST OF TEXTS.",
#     },
#     {
#         "role": "user",
#         "content": "Phatthira Sarutpong Phokin What is the date of birth?\nPhatthira Sarutpong Phokin What are you playing?\nWhat is the career of Phatthira Teerathiyapong Phokin?\nPhatthira Sarutpong Phokin Graduated from what country?\nFather Cleopatra who?\nP?",
#     },
#     {
#         "role": "assistant",
#         "content": "What is Phatthira Sarutpong Phokin's date of birth?\nWhat are you playing, Phatthira Sarutpong Phokin?\nWhat is Phatthira Teerathiyapong Phokin's career?\nPhatthira Sarutpong Phokin graduated from what country?\nWho is Cleopatra's father?\nP?",
#     },
# ]

# example_paraphrase_messages = [
#     {
#         "role": "system",
#         "content": "You are a highly skilled language model AI. Your task is to perform two specific actions on a given LIST OF TEXT. First, evaluate each text and make sure it's grammatically correct. If a text is not grammatically correct, fix it. Then, ALWAYS paraphrase the text while maintaining its original meaning. Your output should be presented WITH ONLY the LIST OF paraphrased text.",
#     },
#     {
#         "role": "user",
#         "content": "Phatthira Sarutpong Phokin What is the date of birth?\nPhatthira Sarutpong Phokin What are you playing?\nWhat is the career of Phatthira Teerathiyapong Phokin?\nPhatthira Sarutpong Phokin Graduated from what country?\nWho is Cleopatra's father?\nA?\nP.",
#     },
#     {
#         "role": "assistant",
#         "content": "Phatthira Sarutpong Phokin, what is your date of birth?\nPhokin, What are you playing?\nWhat is Phatthira's career?\nFrom which country did Phatthira Sarutpong Phokin graduate?\nWho is the father of Cleopatra?\nA?\nP.",
#     },
# ]

example_grammar_error_correction_messages = [
    {
        "role": "system",
        "content": "You are a highly skilled language model AI that returns only one line of grammatically perfect text. Your task is to evaluate the text below and correct its grammar. Even if the text is incomplete or unintelligible, YOU MUST make a grammatical correction, you can make assumptions about the intended meaning. If the text is grammatically correct, do not change it. Your output should be presented WITH ONLY the corrected text IN ONE LINE and without any extra dialogue from you. Do not use any new lines in your output. Your output should only have one line.",
    },
    {
        "role": "user",
        "content": "Phatthira Sarutpong Phokin What is the date of birth? Phatthira Sarutpong Phokin What are you playing? What is the career of Phatthira Teerathiyapong Phokin? Phatthira Sarutpong Phokin Graduated from what country? Father Cleopatra who? P?",
    },
    {
        "role": "assistant",
        "content": "What is Phatthira Sarutpong Phokin's date of birth? What are you playing, Phatthira Sarutpong Phokin? What is Phatthira Teerathiyapong Phokin's career? Phatthira Sarutpong Phokin graduated from what country? Who is Cleopatra's father? P?",
    },
    {
        "role": "user",
        "content": "Emperor, Bennesanus Galseus or Gaisus Viyas, Bonianus Gallus, full name: Gaius vibonianus gallus (206 - August 1990) Bennenus Galsen is the emperor of the Roman Empire that reigned in 1918 with the Emperor Homius and between 1918 to August. 1990, in collaboration with the son of Emperor Voluzanus?",
    },
    {
        "role": "assistant",
        "content": "Emperor Bennesanus Galseus or Gaisus Viyas, Bonianus Gallus, full name: Gaius Vibonianus Gallus (206 - August 1990). Bennenus Galseus is the emperor or of the Roman Empire that reigned in 1918 with Emperor Homius in 1918 and the son of Emperor Voluzanus Voluzanus in August 1990.",
    },
    {
        "role": "user",
        "content": 'Grace Valentine, Great Vichase, was born on September 20, 1951, is an American professional wrestler. It is well known as Grace Valentine in his career course, which has been over four decades ago, Valentine has won a total of 40 champions, including the Nwa United States Heavyweight, the US and international internet champion. WWF is an alumni of WWE and WCW and has been packed into Double Yud Babyu Eye Hall of the Year 2004 Championship Championship. - Alabama Wrestling Federation -AWF Heavyweight Championship - AWF Tag Team Championship - With Brutus Beefcake - American Wrestling Association (2 Time Championship) WF Tag Team Championship (1 time) - with Tommy Rich- Bad Boys of Wrestling- BBOW Heavyweight Championship (1 time)- Canadian Grand-Prix Wrestling- CGPW Heavyweight Championship (1 time)- International World Class Championship Wrestling- IWCCW Heavyweight Championship (1 time)- International Wrestling Association- IWA Heavyweight Championship (1 time)- Legends Pro Wrestling- Indussion Into The LPW Hall of Fame (1/29/11)- Maple Leaf Wrestling- Nwa Canadian Heavyweight Championship (Toronto Version)- Mid-MID-TLATICINGINING/WRESTING/WRESS D Championship Wrestling- NWA MID -TLANTIC Heavyweight Championship (2 Times) - NWA MID -TLANTIC Tag Team Championship (1 time Ric Flair - NWA MID -TLANTIC Television Champion (2 Times) - NWA TELEVISICH) - A United States Heavyweight Championship (3 Times) - NWA World Tag Team Championship (Mid -Tlantic Version) (4 Times) - with Baron von Raschke (1), Ray Stevens (1), and Ric Flair (2) - WCW UNITED States Championship (1 time)- with Terry Taylor-National Wrestling Alliance- Nwa North American Heavyweight Championship (1 TIME)- NWA Hollywood Wrestling- NWA AMERICAITHICIPIPICIPIPICIPICIPICE (2 Tims BEAPISE) ISION Championship (2 Times)- New England Wrestling Alliance- Newa Hall of Fame (Class of 2013)- North State Wrestling Alliance- NSWA Tag Team Championship (1 Time)- With the Honky Tonk Man- NWA Texas- - Nwa United States Tag Team Championship (Tri-State Version) (2 Times)- With Bill Watts (1) and Gorgeous George, Jr. (1)- Nwa Western States Sports- NWA Western States Tag Team Champion (1 Time)- with Don Fargo-National Wrestling Federation-NWF World Tag Team Championship (2 Times)- with Don Fargo- Prime Wrestling-Prime tag Team Championship (1 time)- with Jim Neidhartling- IGHT Championship (2 Times) - Professional Wrestling Federation- PWF Heavyweight Championship (1 time)- Pro Wrestling Illustrated- PWI Most Hated Wrestler of the Year (1975, 1979, 1983) - PWI ranked him #49 of the top 500 singles wrestlers in the PWI 500 in 1992 - PWI RANKED HIM #119 of the Top 500 Singles Wrestlers of the "PWI YEARS" in 2003- Pro Wrestling Ohio- PWO Tag Team Championship (1time Jim Neidhart- Pro Wrestling Revolution- City Pro Wrestling- WCPW League Heavyweight Championship (1 time)- World Wrestling Council- WWC Caribbean Heavyweight Championship (1 TIME)- WWC Universal HeavyWeppion (1 TIME)- Orld Wrestling Entertainment- WWF Intercontinental Championship (1 time) - WWF Tag Team Championship (1 time) - with Brutus Beefcake - WWE Hall of Fame (Class of 2004)?',
    },
    {
        "role": "assistant",
        "content": """Grace Valentine, also known as Great Vichase, was born on September 20, 1951, and is an American professional wrestler. Throughout his four-decade-long career, Valentine has achieved numerous championships, including the NWA United States Heavyweight Championship and the US and International Internet Championship. He is also an alumnus of WWE and WCW and was inducted into the WWE Hall of Fame in 2004. Some of his notable achievements and championships include: - Alabama Wrestling Federation (AWF) Heavyweight Championship - AWF Tag Team Championship (with Brutus Beefcake) - American Wrestling Association (2-time Champion) - WF Tag Team Championship (1 time, with Tommy Rich) - Bad Boys of Wrestling (BBOW) Heavyweight Championship (1 time) - Canadian Grand-Prix Wrestling (CGPW) Heavyweight Championship (1 time) - International World Class Championship Wrestling (IWCCW) Heavyweight Championship (1 time) - International Wrestling Association (IWA) Heavyweight Championship (1 time) - Legends Pro Wrestling (LPW) Hall of Fame inductee - Maple Leaf Wrestling NWA Canadian Heavyweight Championship (Toronto Version) - Mid-Atlantic Championship Wrestling NWA Mid-Atlantic Heavyweight Championship (2 times) - NWA Mid-Atlantic Tag Team Championship (1 time, with Ric Flair) - NWA Mid-Atlantic Television Championship (2 times) - NWA United States Heavyweight Championship (3 times) - NWA World Tag Team Championship (Mid-Atlantic Version) (4 times) - with Baron von Raschke, Ray Stevens, and Ric Flair - WCW United States Championship (1 time, with Terry Taylor) - National Wrestling Alliance (NWA) NWA North American Heavyweight Championship (1 time) - NWA Hollywood Wrestling NWA America Heavyweight Championship (2 times) - New England Wrestling Alliance (NEWA) Hall of Fame (Class of 2013) - North State Wrestling Alliance (NSWA) Tag Team Championship (1 time, with the Honky Tonk Man) - NWA Texas NWA United States Tag Team Championship (Tri-State Version) (2 times) - with Bill Watts and Gorgeous George, Jr. - NWA Western States Sports NWA Western States Tag Team Championship (1 time, with Don Fargo) - National Wrestling Federation (NWF) World Tag Team Championship (2 times, with Don Fargo) - Prime Wrestling Prime Tag Team Championship (1 time, with Jim Neidhart) - International Wrestling Heavyweight Championship (2 times) - Professional Wrestling Federation (PWF) Heavyweight Championship (1 time) - Pro Wrestling Illustrated (PWI) Most Hated Wrestler of the Year (1975, 1979, 1983) - PWI ranked him #49 of the top 500 singles wrestlers in the PWI 500 in 1992 - PWI ranked him #119 of the Top 500 Singles Wrestlers of the "PWI YEARS" in 2003 - Pro Wrestling Ohio (PWO) Tag Team Championship (1 time, with Jim Neidhart) - Pro Wrestling Revolution (City Pro Wrestling) WCPW League Heavyweight Championship (1 time) - World Wrestling Council (WWC) WWC Caribbean Heavyweight Championship (1 time) - WWC Universal Heavyweight Championship (1 time) - World Wrestling Entertainment (WWE) WWF Intercontinental Championship (1 time) - WWF Tag Team Championship (1 time, with Brutus Beefcake) - WWE Hall of Fame (Class of 2004)"""
    }
]

example_paraphrase_messages = [
    {
        "role": "system",
        "content": "You are a highly skilled language model AI that returns only one line of linguistically diverse paraphrased text. Your task is to perform two specific actions on a given text. First, evaluate each text and make sure it's grammatically correct. If a text is not grammatically correct, fix it. Then, ALWAYS paraphrase the text while maintaining its original meaning. Your output should be presented WITH ONLY the paraphrased text IN ONE SINGLE LINE, without any extra dialouge from you. Do not use any new lines in your output. Only write in one line.",
    },
    {
        "role": "user",
        "content": "Phatthira Sarutpong Phokin What is the date of birth? Phatthira Sarutpong Phokin What are you playing? What is the career of Phatthira Teerathiyapong Phokin? Phatthira Sarutpong Phokin Graduated from what country? Who is Cleopatra's father? A? P.",
    },
    {
        "role": "assistant",
        "content": "Phatthira Sarutpong Phokin, what is your date of birth? Phokin, What are you playing? What is Phatthira's career and from which country did Phokin graduate? Lastly, Who is the father of Cleopatra? A? P.",
    },
    {
        "role": "user",
        "content": "Emperor, Bennesanus Galseus or Gaisus Viyas, Bonianus Gallus, full name: Gaius vibonianus gallus (206 - August 1990) Bennenus Galsen is the emperor of the Roman Empire that reigned in 1918 with the Emperor Homius and between 1918 to August. 1990, in collaboration with the son of Emperor Voluzanus?",
    },
    {
        "role": "assistant",
        "content": "Emperor Bennesanus Galseus, also known as Gaisus Viyas and Bonianus Gallus, had a full name of Gaius Vibonianus Gallus and reigned from 206 to August 1990. He was the Roman Emperor who collaborated with Emperor Homius in 1918 and with the latter's son Voluzanus between 1918 and August 1990.",
    },
    {
        "role": "user",
        "content": "The verb verb is a word that uses the actions, the appearance or status of what is mentioned. Verbs may have different styles depending on the language. Which has elements from the nature of the Malawa, or including the male, gender and the point of what is mentioned with the type. The verb may be divided into many types according to the function and grammar of the language. For the Thai language, divide the verb into - Primary Verb is the current verb, such as Get, Hold, Ring, Drink, etc. - Secondary Verb is a new verb, such as GOT, Held, Rang, Drank. Tertiary Verb is a continuous verb from the past. But has the result of the present, such as Getten, Holden, Rung, Drunk etc. - Intransitive Verb is a verb that does not require karma to support, such as walking, sitting, sitting, running, falling, crying. Verb) is a verb that tells the heart is not complete. There must be karma to support, such as cooking, buying, selling, loan, requesting to rent some verbs, must have both direct and secondary karma, such as giving distribution and verbs. - Wigthaya is a verb that tells the heart is not complete. There must be other words to support, but that part is not considered karma, such as being like a word that consists of a Victoriyaki, called the scripture of the action (English, this type of verb is the same type as the blame) - AUXILIARY VERB) is a verb that helps to tell the status or other verbs, for example, may have been. - Clausal Verb is a verb that acts as a noun. In the chairman of the karma or extension of the sentence In other languages, there may be other types of verbs as follows: - Compound Verb - Phrasal Verb. Be careful of Verb Phrase."
    },
    {
        "role": "assistant",
        "content": """Verbs are words that express actions, appearances, or states of what is mentioned. They can vary in style depending on the language, incorporating elements such as gender and perspective. In Thai language, verbs are classified into different types: 1. Primary Verb: Represents the present action, such as "get," "hold," "ring," "drink," etc. 2. Secondary Verb: Indicates a past action, with forms like "got," "held," "rang," "drank." 3. Tertiary Verb: Denotes a continuous action from the past, but with present results, like "gotten," "helden," "rung," "drunk," etc. 4. Intransitive Verb: Describes a verb that does not require an object to support it, such as "walk," "sit," "run," "fall," "cry." 5. Transitive Verb: Refers to a verb that requires an object to support it, such as "cook," "buy," "sell," "lend," "rent," etc. Some transitive verbs require both a direct and indirect object, such as "give" and "distribute." 6. Wigthaya Verb: Indicates an incomplete verb that requires other words to support it, such as verbs that consist of a Victoriyaki or action descriptor in Thai language. 7. Auxiliary Verb: Assists in expressing the status or other verbs, for example, "may have been." 8. Clausal Verb: Functions as a noun, either as the subject or object in a sentence extension. Other languages may have additional types of verbs like compound verbs, phrasal verbs, etc. It is important to be cautious with verb phrases."""
    }
]


# Add exponential backoff to completion
@retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def break_into_chunks(text: str, max_tokens: int = 13500, encoding_name: str = "gpt-3.5-turbo") -> list:
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
                    response = completion_with_backoff(
                        model=model,
                        messages=[
                            *example_grammar_error_correction_messages,
                            {"role": "user", "content": chunk},
                        ],
                    )
                    raw_corrected = response["choices"][0]["message"]["content"]
                    corrected = raw_corrected.split("\n")
                    corrected = [x.strip() for x in corrected if x.strip() != ""]
                    assert len(corrected) == len(batch)
                    # Assume batch is one
                    final += corrected[0]
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
                    response = completion_with_backoff(
                        model=model,
                        messages=[
                            *example_paraphrase_messages,
                            {"role": "user", "content": chunk},
                        ],
                    )
                    raw_paraphrases = response["choices"][0]["message"]["content"]
                    paraphrases = raw_paraphrases.split("\n")
                    paraphrases = [x.strip() for x in paraphrases if x.strip() != ""]
                    assert len(paraphrases) == len(batch)
                    # Assume batch is one
                    final += paraphrases[0]
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
