from typing import NamedTuple

import openai

from evalio import util

CORRECT_GRAMMAR_TEMPLATE = (
    'Is the following text grammatically correct? Ignore style and meaning, just strictly check grammar.\nIf it is'
    " grammatically correct, just respond with 'yes', nothing more.\nIf it is grammatically incorrect, respond with"
    ' the corrected text. Do not explain what you did or why, just correct it.\n\nText:\n"""\n{text}\n"""\nResponse:\n'
)


class GrammarCorrection(NamedTuple):
    is_correct: bool
    correction: str | None


def is_correct_grammar(text) -> GrammarCorrection:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are GrammarAndWritingAssistant, a helpful AI Assistant."},
            {"role": "user", "content": CORRECT_GRAMMAR_TEMPLATE.format(text=text)},
        ],
        temperature=0,
    )
    correction = None
    response_text = util.unquote(response.choices[0].message["content"]).strip()
    is_correct = response_text.lower() in ("yes", "yes.")
    if not is_correct:
        correction = response_text
    return GrammarCorrection(is_correct=is_correct, correction=correction)
