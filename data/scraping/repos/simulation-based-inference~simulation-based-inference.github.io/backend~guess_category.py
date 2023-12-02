import openai
import os
import re
import json
import logging
from backend.database import get_papers, Paper, write_papers
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Tuple, List

load_dotenv()


def post_process_guesses(guess: str) -> str:
    """Post-process guesses."""

    guess = guess.lower()
    if "uncertain" in guess:
        return None

    # remove prefix
    guess = guess.removeprefix("subject category: ")
    guess = guess.removeprefix("subject categories: ")
    guess = guess.removeprefix("category: ")
    guess = guess.removeprefix("categories: ")

    # remove suffix
    guess = guess.removesuffix(".")

    # Remove secondary categories
    guess = guess.split("/")[0]
    guess = guess.split(",")[0]
    guess = guess.split(";")[0]
    guess = guess.split("and")[0]
    guess = guess.split("&")[0]

    # Remove brackets
    guess = re.sub(r"\(.*?\)", "", guess)

    return guess.strip()


def guess_category(title: str) -> str:
    """Guess category with OpenAI."""

    openai.api_key = os.getenv("OPENAI_API_KEY")
    instruction = "Given a journal article title, predict a subject category. Only provide one category, if you are not sure, just return 'uncertain'."
    prompt = f"Title: {title}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ],
    )
    return post_process_guesses(response.choices[0].message.content)


class Guesser:
    """Guess category with OpenAI and save intermediate guesses."""

    GUESS_JSON = "backend/data/guess_category_inference.json"
    GROUP_JSON = "backend/data/guess_category_group.json"
    OVERRIDE_JSON = "backend/data/category_override.json"

    def __init__(self):
        with open(self.GROUP_JSON, "r") as f:
            self.category_group = json.load(f)

        with open(self.GUESS_JSON, "r") as f:
            self.guesses = json.load(f)

        with open(self.OVERRIDE_JSON, "r") as f:
            self.override = json.load(f)

    def guess(self, id: int, title: str) -> str | None:
        """Guess and log the category group."""

        id = str(id)
        if id in self.guesses:
            logging.info("Using cached guess")
            guess = self.guesses[id]
        else:
            guess = guess_category(title)
            self.guesses[id] = guess
            with open(self.GUESS_JSON, "w") as f:
                json.dump(self.guesses, f, indent=4)

        if guess is None:
            return None

        if guess not in self.category_group:
            logging.warning("Adding new category group")
            # add to group and maybe update the group name manually later
            self.category_group[guess] = guess
            with open(self.GROUP_JSON, "w") as f:
                json.dump(self.category_group, f, indent=4)

        return self.category_group[guess]

    def regenerate_categories(self, papers: List[Paper]) -> None:
        """Regenerate all guessed categories (for updating manual group label)."""

        for paper in tqdm(papers):
            # Top priority is the override
            if str(paper.id) in self.override:
                paper.category = self.override[str(paper.id)]
                continue

            # Then, guess if the category is not already set
            if str(paper.id) in self.guesses:
                if self.guesses[str(paper.id)] is None:
                    continue
                paper.category = self.guess(paper.id, paper.title)

        write_papers(papers)


def _test_guesses() -> Tuple[dict, dict, dict]:
    """Test the guess_category function."""

    results = {}
    titles = {}
    real_categories = {}

    papers = get_papers(as_dict=True)
    test_papers = [paper for paper in papers if "category" in paper]
    for paper in tqdm(test_papers):
        try:
            real_categories[paper["id"]] = paper["category"]
            results[paper["id"]] = (
                guess_category(paper["title"])
                .removeprefix("subject category: ")
                .removeprefix("subject categories: ")
                .removesuffix(".")
            )
            titles[paper["id"]] = paper["title"]
        except Exception as e:
            print(e)
            print(paper["title"])
            pass

    return results, titles, real_categories
