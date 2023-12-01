#!/usr/bin/env python3

import openai
from tqdm import tqdm
import os
import sys

from collections import defaultdict
from typing import List, Union, Dict
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode
import gender_guesser.detector as gender_guesser


def get_authors_bibtex(filename: str) -> List[str]:
    with open(filename, "r") as bibtex_file:
        bib_db = BibTexParser(
            bibtex_file.read(), customization=convert_to_unicode
        )  # .get_database()
    author_list = []
    for entry in bib_db.entries:
        authors = entry.get("author", "").split(" and ")
        author_list.extend(authors)

    print("#Papers:", len(bib_db.entries))
    print("#Authors:", len(author_list))
    print("#Authors_unique:", len(list(set(author_list))))
    print()
    return author_list


def process_gender_stats(authors: List[str]) -> Union[int, float]:
    d = gender_guesser.Detector(case_sensitive=False)
    male, female, andy, unknown = 0, 0, 0, 0
    first_names = set()
    name_count = defaultdict(int)
    guesses = dict()
    for author in authors:
        if len(author) == 0:
            continue
        if len(author.split()) > 1:
            first_name = author.split()[1]
        else:
            first_name = author.split()[0]

        gender = d.get_gender(first_name)

        first_names.add(first_name)
        name_count[first_name] += 1
        if "female" in gender:
            female += 1
            guesses[first_name] = "F"
        elif "male" in gender:
            male += 1
            guesses[first_name] = "M"
        elif "andy" in gender:
            andy += 1
            guesses[first_name] = "A"
        else:
            unknown += 1
            guesses[first_name] = "U"
    print(f"Number of authors: {len(authors)}")
    print(f"Number of estimated men: {male}")
    print(f"Number of estimated women: {female}")
    print(f"Number of estimated ambiguous: {andy}")
    print(f"Number of estimated unknown: {unknown}")
    print(f"Male-Female ratio: {male/female if female else 'Inf'}")

    guesses_str = ""
    for k, v in guesses.items():
        guesses_str += f"{k}: {v}\n"
    with open("/tmp/gender_guess.txt", "w") as f:
        f.write(guesses_str)
    return list(first_names), name_count, guesses


def query_response(messages, model, num_names):

    # completion = openai.Completion.create(
    #     model=model,
    #     prompt=messages,
    # )
    # out = completion.choices[0].text
    # completion = openai.ChatCompletion.create(
    #     model=model,
    #     messages=messages,
    # )
    # out = completion.choices[0].message.content

    # for some reason, at time of developing, this only worked when
    # I streamed the output data.
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
        stream=True,
    )
    completion_text = ""

    with tqdm(total=num_names) as pbar:
        for chunk in response:
            try:
                event_text = chunk["choices"][0]["delta"]["content"]
            except Exception as e:
                if chunk["choices"][0]["finish_reason"] == "stop":
                    break
                else:
                    raise e
            completion_text += event_text
            if "\n" in event_text:
                pbar.update(1)
    return completion_text


def llm_based_gender_estimate(
    first_names: List[str], name_count: Dict[str, int], bib_file: os.PathLike
):
    model = "gpt-3.5-turbo-16k"
    # model = "gpt-4"
    # model = "text-davinci-003"
    message = """Below lists the first names for the authors of research papers.
    For each, guess the most likely gender given the name.
    M for male, F for female, A for ambigous, U for unknown.
    Return data in the format, with no additional text:

    Jack: M
    Susan: F
    Pauley: A
    PyTorch: U

    The next message includes the list:
    """

    bib_file = bib_file.replace("..", "")
    bib_file = bib_file.replace("/", "")
    cache_file = f"/tmp/llm_gender_guess_{bib_file}.txt"

    # LLM inference is expensive, check if we've run it before
    if not os.path.exists(cache_file):
        messages = []
        messages.append({"role": "system", "content": message})
        messages.append({"role": "user", "content": "\n".join(first_names)})

        out = query_response(messages, model, len(first_names))

        guesses = out.split("\n")

        with open(cache_file, "w") as f:
            f.write(out)
    else:
        with open(cache_file, "r") as file:
            guesses = file.readlines()

    male, female, andy, unknown = 0, 0, 0, 0
    llm_guesses = dict()

    for g in guesses:
        # this means we reduce the number of LLM tokens
        name, gender_guess = g.split(": ")
        name, gender_guess = name.strip(), gender_guess.strip()
        if gender_guess == "M":
            male += name_count[name]
            llm_guesses[name] = "M"
        elif gender_guess == "F":
            female += name_count[name]
            llm_guesses[name] = "F"
        elif gender_guess == "A":
            andy += name_count[name]
            llm_guesses[name] = "A"
        else:
            unknown += name_count[name]
            llm_guesses[name] = "U"

    print(f"Number of estimated men: {male}")
    print(f"Number of estimated women: {female}")
    print(f"Male-Female ratio: {male/female if female else 'Inf'}")
    print(f"Number of estimated ambiguous: {andy}")
    print(f"Number of estimated unknown: {unknown}")
    print(f"Male-Female ratio: {male/female if female else 'Inf'}")
    return llm_guesses


def find_disagreements(guess, llm_guesses):
    disagreements = {}
    for key in guess:
        if key in llm_guesses and guess[key] != llm_guesses[key]:
            disagreements[key] = {"guess": guess[key], "llm": llm_guesses[key]}
    return disagreements


def main(bib_file):
    authors = get_authors_bibtex(bib_file)
    print("# gender-guesser mode")
    first_names, name_count, guesses = process_gender_stats(authors)
    print()
    print("# LLM mode")
    llm_guesses = llm_based_gender_estimate(first_names, name_count, bib_file)

    # print(find_disagreements(guesses, llm_guesses))


if __name__ == "__main__":
    """Estimate the gender of authors in your bibliography.
    Uses two methods:
    1. one using the gender-guesser package (outdated, somewhat Euro-centric)
    2. LLM using the OpenAI API, which may be more accurate
    Ambiguity is somewhat handled, but assumptions are still made.
    Consider this an unanuanced estimate.

    To generate a sub-bibliography of papers you _actually_ cite from your bibfile
    run (assuming your main tex file is `00-main.tex` and main bib is `refs.bib`).
    Generates a new bib file `extracted.bib`:

    ```sh
    pdflatex -shell-escape 00-main.tex
    biber 00-main
    pdflatex -shell-escape 00-main.tex
    pdflatex -shell-escape 00-main.tex

    jabref -a 00-main,/tmp/extracted refs.bib
    ```
    """
    if len(sys.argv) < 2:
        print("Usage: python bib_stats.py bibtex_file")
    else:
        bibtex_file = sys.argv[1]
        main(bibtex_file)
