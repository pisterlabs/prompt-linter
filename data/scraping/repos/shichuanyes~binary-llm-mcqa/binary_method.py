from typing import List

import openai

from method.method import Method
from question import Question


class BinaryMethod(Method):
    @staticmethod
    def get_prompts(question: Question) -> List[str]:
        prompt = f"{question.get_parts_str()}\n"
        result: List[str] = []
        for choice in question.choices:
            result.append(
                prompt
                + f"Answer: {choice}\n"
                + "Correct:"
            )
        return result

    @staticmethod
    def get_completions(question: Question) -> List[str]:
        return [' yes' if idx == question.answer_idx else ' no' for idx in range(len(question))]

    @classmethod
    def ask(cls, question: Question, model: str) -> int:
        prompts = cls.get_prompts(question)

        result: int = -1
        max_: float = -1e3  # probs smaller than that is probably just guessing
        for i, prompt in enumerate(prompts):
            res = openai.Completion.create(
                model=model,
                prompt=prompt,
                max_tokens=1,
                temperature=0,
                logprobs=2
            )
            top_logprobs = res['choices'][0]['logprobs']['top_logprobs'][0]
            if ' yes' in top_logprobs and top_logprobs[' yes'] > max_:
                max_ = top_logprobs[' yes']
                result = i
        return result
