from typing import List
from letter_state import LetterState
from wordle import Wordle
from colorama import Fore
import random
import openai
import os

def main():

    word_set = load_word_set("data/wordle_words.txt")
    secret = random.choice(list(word_set))
    wordle = Wordle(secret)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    enriched_prompt = f"Generate a simplified definition for {wordle.secret}: "


    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=enriched_prompt
    )


    # Extract output text.
    definition: str = response["choices"][0]["text"]

    while wordle.can_attempt:
        x = input("\nType your guess: ")

        if len(x) != wordle.WORD_LENGTH:
            print(
                Fore.RED
                + f"Word must be {wordle.WORD_LENGTH} characters long!"
                + Fore.RESET
            )
            continue

        if not x in word_set:
            print(
                Fore.RED
                + f"{x} is not a valid word!"
                + Fore.RESET
            )
            continue

        wordle.attempt(x)
        display_results(wordle)

    if wordle.is_solved:
        print(f"You've solved the puzzle, the word means: {definition}")
    else:
        print("You failed to solve the puzzle!")
        print(f"The secret word was: {wordle.secret}, which means: {definition}")


def display_results(wordle: Wordle):
    print("\nYour results so far...")
    print(f"You have {wordle.remaining_attempts} attempts remaining.\n")

    lines = []

    for word in wordle.attempts:
        result = wordle.guess(word)
        colored_result_str = convert_result_to_color(result)
        lines.append(colored_result_str)

    for _ in range(wordle.remaining_attempts):
        lines.append(" ".join(["_"] * wordle.WORD_LENGTH))

    draw_border_around(lines)


def load_word_set(path: str):
    word_set = set()
    with open(path, "r") as f:
        for line in f.readlines():
            word = line.strip().upper()
            word_set.add(word)
    return word_set


def convert_result_to_color(result: List[LetterState]):
    result_with_color = []
    for letter in result:
        if letter.is_in_position:
            color = Fore.GREEN
        elif letter.is_in_word:
            color = Fore.YELLOW
        else:
            color = Fore.WHITE
        colored_letter = color + letter.character + Fore.RESET
        result_with_color.append(colored_letter)
    return " ".join(result_with_color)


def draw_border_around(lines: List[str], size: int = 9, pad: int = 1):

    """Draws a border around a list of strings."""
    
    # Calculate the length of the content including padding
    content_length = size + 2 * pad
    
    # Define the top and bottom borders
    top_border = "┌" + "─" * content_length + "┐"
    bottom_border = "└" + "─" * content_length + "┘"
    
    # Create a padding space string
    padding_space = " " * pad
    
    # Print the top border
    print(top_border)
    
    # Print each line with side borders and padding
    for line in lines:
        print(f"│{padding_space}{line}{padding_space}│")
    
    # Print the bottom border
    print(bottom_border)


if __name__ == "__main__":
    main()
