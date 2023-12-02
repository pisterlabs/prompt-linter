import time
from dataclasses import dataclass

import openai
from loguru import logger
from mini_judge.judge_prompt import judge_prompt_template

MAX_TOKENS = 2048
API_ERROR_OUTPUT = "API_ERROR"
API_MAX_RETRY = 3
API_RETRY_SLEEP = 10


@dataclass
class TrialOutcome:
    judge_answer: str
    trial_verdict: int


def parse_judge_answer(judge_answer):
    if judge_answer.endswith('[[B]]'):
        return -1
    elif judge_answer.endswith('[[A]]'):
        return 1
    return 0


def trial(judge_model: str, question: str, candidate_answer: str, ref_answer: str) -> TrialOutcome:
    prompt = judge_prompt_template.format(question=question, candidate_answer=candidate_answer,
                                          ref_answer=ref_answer)
    messages = format_openai_chat_prompt(prompt)
    judge_answer = chat_completion_openai(judge_model, messages,
                                          temperature=0, max_tokens=MAX_TOKENS)

    trial_verdict = parse_judge_answer(judge_answer)
    return TrialOutcome(judge_answer, trial_verdict)


def format_openai_chat_prompt(prompt_text: str):
    return [{'role': 'user', 'content': prompt_text}]


def chat_completion_openai(model, messages, temperature, max_tokens):
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response["choices"][0]["message"]["content"]
            break
        except openai.error.OpenAIError as e:
            logger.error(e)
            time.sleep(API_RETRY_SLEEP)

    return output
