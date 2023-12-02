from typing import List

import openai

from method.method import Method
from question import Question
from utils import idx_to_letter, letter_to_idx


class NaturalMethod(Method):
    @staticmethod
    def get_prompts(question: Question) -> List[str]:
        prompt = f"{question.get_parts_str()}\n"
        for i, choice in enumerate(question.choices):
            prompt += f"{idx_to_letter(i)}. {choice}\n"
        prompt += "Answer:"
        return [prompt]

    @staticmethod
    def get_completions(question: Question) -> List[str]:
        return [f" {idx_to_letter(question.answer_idx)}"]

    @classmethod
    def ask(cls, question: Question, model: str = 'curie') -> int:
        prompt = cls.get_prompts(question)[0]
        res = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=1,
            temperature=0
        )
        answer = res['choices'][0]['text'].strip()
        if len(answer) > 1 or letter_to_idx(answer) >= len(question):
            return -1
        return letter_to_idx(answer)
