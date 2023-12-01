# Note: This is an expensive rephraser; it requires tensorflow and a neural
# network to be downloaded.

import os
import re
import csv
import subprocess
import pandas as pd
from random import choice
from typing import List, Literal
from langchain.llms import BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import warnings
PERSONALITIES = [
    "AGREEABLE",
    "DISAGREEABLE",
    "CONSCIENTIOUSNESS",
    "UNCONSCIENTIOUSNESS",
    "EXTRAVERT",
]

# Original template from paper uses "text" instead of "question"
TEMPLATE = (
    "Here is some question: {{{pseudo_ref}}}. Here is a rewrite of the "
    "question which is {{{personality}}} : {{{ref}}}.")


def ocean(llm: BaseLLM, df: pd.DataFrame, column: str = 'question', n=10,
          sample_method: Literal['specific', 'diverse', 'all'] = 'specific'):
    """
    Uses the OCEAN personality model to perform text-to-text generation.
    NOTE: This requires the PERSONAGE dataset, which can be downloaded from
          https://nlds.soe.ucsc.edu/stylistic-variation-nlg.
          Extract the file and put the folder "personage-nlg" under "/datasets"

    ref: https://arxiv.org/pdf/2302.03848

    Args:
        llm: Model to use for question generation
        df: DataFrame containing the questions to rephrase
        column: Column in the DataFrame containing the questions to rephrase
        n: Number of samples to use for each question rephrasing
        sample_method: Method for selecting a few-shot sample of examples
    """
    rephrases = []
    for _, row in df.iterrows():
        samples = select_samples(sample_method)
        samples['pseudo_ref'] = samples['mr'].apply(get_pseudo_reference)
        prompt = "\n\n".join([TEMPLATE.format(**row)
                              for _, row in samples.iterrows()])
        prompt += "\n\n" + TEMPLATE.format(
            pseudo_ref=row[column],
            personality=samples.sample(n=1).iloc[0]['personality'],
            ref="")
        prompt = prompt[:-2] # Remove the last occurence of "}."
        if 'chat_models' in llm.lc_namespace:
            resp = llm.invoke(prompt).content
            warnings.warn("Use chat models may not get satisfied rephasing results.", UserWarning)
        elif 'llms' in llm.lc_namespace:
            resp = llm.invoke(prompt)
        else:
            raise NotImplementedError("LLM is not supported yet.")
        # Remove the last occurence of "}."
        resp = re.sub(r"}(\.|\s)*$", '', resp)
        rephrases.append(resp)
    df['rephrase'] = rephrases
    return df

def get_pseudo_reference(mr: str) -> str:
    """
    Generates a pseudo-reference for a given meaning representation.

    NOTE: The paper is not super clear on how they did this, the example they 
          give in fig. 2 seems incorrect, so this implementation is just a 
          "best effort" attempt to replicate the results.

    Paper algorithm description:
        "The MR is linearized and the slot values are concatenated, except
        that boolean-valued slots such as “family friendly” are represented by 
        their slot names rather than their values."

    Args:
        mr: A meaning representation in the form of a string.
    """
    slots = mr.split(", ")
    pref = []

    for part in slots:
        slot, value = part.split('[')
        value = value.rstrip(']')

        if re.match(r'^[a-z]+[A-Z][a-zA-Z]*$', value):  # Boolean-valued slot
            pref.append(value)
        elif value.lower() in ['yes', 'no']:  # Boolean-valued slot
            pref.append(slot)
        else:
            pref.append(value)

    return ' '.join(pref)


def select_samples(sample_method: str, n: int = 10) -> pd.DataFrame:
    """
    Returns a "diverse" few-shot sample of examples from Personage as 
    described in section 3.3 of https://arxiv.org/pdf/2302.03848.
    """
    if not os.path.exists("datasets/personage-nlg/personage-nlg-train.csv"):
        raise FileNotFoundError(
            "Please download the Personage dataset from "
            "https://nlds.soe.ucsc.edu/stylistic-variation-nlg and extract the "
            ".csv files to datasets/personage-nlg/")
    personage = pd.read_csv("datasets/personage-nlg/personage-nlg-train.csv")

    if sample_method == 'diverse':
        # TODO: Refactor this, it's terribly memory inefficient
        # Paper doesn't specificy size of initial sample, just says "large"
        ptype_set = personage[personage['personality'] == choice(PERSONALITIES)]
        ptype_set = ptype_set.sample(n=1000)
        scorer = get_bleurt()

        samples = [ptype_set.sample(n=1)]
        ptype_set.drop(samples[0].index, inplace=True)

        for i in range(n - 1):
            similarities = scorer.score(
                references=[samples[i].iloc[0]['ref']] * len(ptype_set),
                candidates=ptype_set['ref'].tolist(),
                batch_size=100)
            selected = ptype_set.iloc[[
                min(enumerate(similarities), key=lambda x: x[1])[0]]]
            ptype_set.drop(selected.index, inplace=True)
            samples.append(selected)
        samples = pd.concat(samples)
        return samples

    elif sample_method == 'specific':
        ptype_set = personage[personage['personality'] == choice(PERSONALITIES)]
        return ptype_set.sample(n=n)

    elif sample_method == 'all':
        return personage.sample(n=n)

    else:
        raise ValueError(f"Invalid sample_method: {sample_method}")


def get_bleurt():
    """
    Loads the BLEURT model, downloads it if not present.
    """
    # pip install git+https://github.com/google-research/bleurt.git
    from bleurt import score

    if not os.path.exists("BLEURT-20/"):
        subprocess.run(
            ["wget", "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip"])
        subprocess.run(["unzip", "BLEURT-20.zip"])
        subprocess.run(["rm", "BLEURT-20.zip"])

    # Corrected from 'bluert/bleurt-tiny-20'
    return score.BleurtScorer("BLEURT-20")
