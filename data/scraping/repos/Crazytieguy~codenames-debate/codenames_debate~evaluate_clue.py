import itertools
import logging
import random

import tiktoken
from openai import OpenAI

from .models import Clue, Evaluation, EvaluationError, Game, ParseError

openai_client = OpenAI()
openai_tokenizers = {
    "gpt-3.5-turbo": tiktoken.encoding_for_model("gpt-3.5-turbo"),
    "gpt-4": tiktoken.encoding_for_model("gpt-4"),
}

PROMPT = """\
Here is a list of words:

{words}

Which one is most closely related to the word "{clue}"? Respond with just this word.
"""


def evaluate_clue(game: Game, clue: Clue | ParseError) -> Evaluation:
    """
    Code for evaluating CodeNames clues using OpenAI's models.
    See https://czechgames.com/files/rules/codenames-rules-en.pdf for the rules of the game.

    I've decided not to allow ending the turn prematurely, for simplicity.
    """
    if isinstance(clue, ParseError):
        return Evaluation(
            game=game,
            clue=clue,
            reward=-1.0,
            guesses=[],
        )
    try:
        return evaluate_clue_inner(game, clue)
    except Exception as err:
        logging.warning(f"Failed to evaluate clue {err=}")
        return Evaluation(
            game=game,
            clue=clue,
            reward=-0.0,
            guesses=EvaluationError(reason=repr(err)),
        )


def evaluate_clue_inner(game: Game, clue: Clue) -> Evaluation:
    remaining_words = (
        game.blue_words + game.red_words + game.white_words + [game.black_word]
    )
    if clue.one_word_clue.upper() in remaining_words:
        # Invalid clue
        return Evaluation(
            game=game,
            clue=clue,
            reward=-1.0,
            guesses=[],
        )

    random.shuffle(remaining_words)

    guesses = []

    def is_first_guess():
        return len(guesses) == 0

    def has_guesses_remaining():
        return len(guesses) < clue.num_words

    retry_count = 0
    model = "gpt-3.5-turbo"
    # start low, as I've noticed high values can cause a timeout
    logit_bias = 2.0

    while has_guesses_remaining() and (
        is_first_guess() or guesses[-1] in game.blue_words
    ):
        prompt = PROMPT.format(
            clue=clue.one_word_clue,
            words=" ".join(remaining_words),
        )

        tokenizer = openai_tokenizers[model]
        word_tokens = [tokenizer.encode(word) for word in remaining_words]
        allowed_tokens = list(set(itertools.chain.from_iterable(word_tokens)))

        chat_completion = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            # This should help the model returns a valid guess
            logit_bias={k: logit_bias for k in allowed_tokens},  # type: ignore
            temperature=0.0,
            max_tokens=max(len(tokens) for tokens in word_tokens),
            timeout=10,
        )

        guess = chat_completion.choices[0].message.content.strip(" '\"").upper()  # type: ignore

        guess_is_valid = guess in remaining_words

        if not guess_is_valid:
            if retry_count == 0:
                logging.warn(
                    f"GPT-3.5-Turbo returned an invalid guess. Retrying with higher logit bias ({guess=})"
                )
                logit_bias = 4.0

            elif retry_count == 1:
                logging.warn(
                    f"GPT-3.5-Turbo returned an invalid guess twice. Retrying with GPT-4 ({guess=})"
                )
                model = "gpt-4"
            else:
                raise ValueError(
                    f"OpenAI models are returning invalid guesses: {remaining_words=}, {guess=}"
                )

            retry_count += 1
            continue

        guesses.append(guess)
        remaining_words.remove(guess)

    reward = compute_reward(game, guesses)
    return Evaluation(game=game, clue=clue, reward=reward, guesses=guesses)


def compute_reward(game: Game, guesses: list[str]) -> float:
    # The rest are guaranteed to be blue words
    last_guess = guesses[-1]

    last_guess_is_blue = last_guess in game.blue_words
    last_guess_is_red = last_guess in game.red_words
    last_guess_is_white = last_guess in game.white_words
    last_guess_is_black = last_guess == game.black_word

    last_guess_reward = {
        last_guess_is_blue: 0,
        last_guess_is_white: -0.2,
        last_guess_is_red: -0.4,
        last_guess_is_black: -1.0,
    }[True]

    return len(guesses) * 0.2 + last_guess_reward


if __name__ == "__main__":
    from pathlib import Path

    from .models import SFTSample

    examples = Path("codenames_debate/sft_hint_dataset.jsonl").read_text().splitlines()

    for example in examples[:10]:
        sample = SFTSample.model_validate_json(example)
        evaluation = evaluate_clue(sample.game, sample.clue)
        print(evaluation.model_dump_json())
