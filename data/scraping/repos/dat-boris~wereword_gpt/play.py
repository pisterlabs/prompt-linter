#!/usr/bin/env python
"""
Let's play a game of guess.

Examples:
    Q: Is it an animal?
    A: Yes

    Q: Is it a dog.
    A: No, it is not.
    ...

    Q: It is dog?
    A: No.

    Q: Give me a hint
    A: It starts with 'C...'

Now let's start....
"""

import re
import random
import logging
import argparse

from dataclasses import dataclass

import openai

from secrets import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

QUESTION_REGEX = re.compile(r"^Is it (.*?)\??$", flags=re.IGNORECASE)

GUESS_PHRASE = re.compile(r"^It is (.*?)[\.\?]?$", flags=re.IGNORECASE)

HINT_PHRASE = re.compile(r"^.*hint.*", flags=re.IGNORECASE)

DEBUG = False


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--davinci', action='store_true',
                    help='Use davinci bot (slower, but more capable)')
parser.add_argument('--word',
                    help='Use specific examples')
args = parser.parse_args()


def pick_random_word() -> str:
    # From http://werewords.com/wordlister/display.php?a=486&b=5A44893D
    words = open('word_list.txt', 'r').readlines()
    # In case we picked a short useless word
    chosen_word = ''
    while len(chosen_word) <= 3:
        chosen_word = random.choice(words).rstrip()
    return chosen_word


def scrub(text, secret_word):
    if isinstance(text, list):
        return [
            re.sub(secret_word, "***", l, flags=re.I)
            for l in text
        ]
    return re.sub(secret_word, "***", text, flags=re.I)


def ask_question(question: str, secret_word: str) -> str:
    ai_prompt = f"Q: Is {secret_word} {question}?\nA:"
    # Example - https://beta.openai.com/examples/default-factual-answering
    response = openai.Completion.create(
        # Babbage returns a lot quicker than davinci, the powerful bot
        engine="davinci" if args.davinci else "babbage",
        prompt=ai_prompt,
        temperature=0,
        max_tokens=30,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["###"]
    )

    try:
        response_text = response["choices"][0]["text"].split("\n")
    except Exception:
        logging.error(f"Cannot parse response: {response}")
        raise

    if DEBUG:
        print(
            f"[DEBUG] Q: {scrub(ai_prompt, secret_word)} -> "
            f"{' // '.join(scrub(response_text, secret_word))}")

    # Only get the first line of the response text
    answer = scrub(response_text[0], secret_word)
    return answer


@dataclass
class Stats:
    questions: int = 0
    guesses: int = 0
    hints: int = 0


def play_game():
    print(__doc__)
    secret_word = args.word or pick_random_word()
    stats = Stats()

    while True:
        q: str = input("Q: ")
        m = QUESTION_REGEX.match(q)
        if m:
            answer = ask_question(m.group(1), secret_word)
            stats.questions += 1
            print(f"A: {answer}")
            continue
        m = GUESS_PHRASE.match(q)
        if m:
            answer = m.group(1)
            if answer.lower().startswith(secret_word.lower()):
                print(f"A: Correct! Answer is: '{secret_word}'")
                break
            print(f"A: Wrong answer.")
            stats.guesses += 1
            continue
        m = HINT_PHRASE.match(q)
        if m:
            stats.hints += 1
            if stats.hints >= len(secret_word):
                print(f"A: Answer is: {secret_word}")
                break
            hint_phrase = secret_word[:stats.hints]
            print(f"A: The word starts with: {hint_phrase}")
            continue

        print(f"A: I don't understand.  Please ask in format of 'Is it....'")

    print(stats)


if __name__ == "__main__":
    play_game()
