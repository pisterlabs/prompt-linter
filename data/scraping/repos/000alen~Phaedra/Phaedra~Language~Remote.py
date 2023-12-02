from typing import List

import openai
import transformers  # type: ignore

from Phaedra.Language.Base import (
    summarizer_parameters,
    summarizer_prompt,
    answerer_parameters,
    answerer_prompt,
    generator_parameters,
    generator_prompt,
)

_tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")


def get_summarizer_tokenizer():
    """Returns the summarizer tokenizer.

    :return: The summarizer tokenizer.

    """

    return _tokenizer


def get_answerer_tokenizer():
    """Returns the answerer tokenizer.

    :return: The answerer tokenizer.

    """

    return _tokenizer


def get_generator_tokenizer():
    """Returns the generator tokenizer.

    :return: The generator tokenizer.

    """

    return _tokenizer


def summarize(text: str) -> str:
    """Summarizes the text (remote mode).

    :param text: The text to summarize.
    :type text: str
    :return: The summarized text.
    :rtype: str

    """

    parameters = summarizer_parameters
    prompt = summarizer_prompt.format(text=text)

    response = openai.Completion.create(**parameters, prompt=prompt)

    return response["choices"][0]["text"]


def batch_summarize(texts: List[str]) -> List[str]:
    """Summarizes the texts (remote mode).

    :param texts: The texts to summarize.
    :type texts: List[str]
    :return: The summarized texts.
    :rtype: List[str]

    """

    parameters = summarizer_parameters
    prompts = [summarizer_prompt.format(text=text) for text in texts]

    response = openai.Completion.create(**parameters, prompt=prompts)

    return [choice["text"] for choice in response["choices"]]


def answer(question: str, context: str) -> str:
    """Answers the question with the given context (remote mode).

    :param question: The question to answer.
    :type question: str
    :param context: The context to answer the question with.
    :type context: str
    :return: The answer.
    :rtype: str

    """

    parameters = answerer_parameters
    prompt = answerer_prompt.format(question=question, context=context)

    response = openai.Completion.create(**parameters, prompt=prompt)

    return response["choices"][0]["text"]


def batch_answer(questions: List[str], contexts: List[str]) -> List[str]:
    """Answers the questions with the given contexts (remote mode).

    :param questions: The questions to answer.
    :type questions: List[str]
    :param contexts: The contexts to answer the questions with.
    :type contexts: List[str]
    :return: The answers.
    :rtype: List[str]

    """

    parameters = answerer_parameters
    prompts = [
        answerer_prompt.format(question=question, context=context)
        for question, context in zip(questions, contexts)
    ]

    response = openai.Completion.create(**parameters, prompt=prompts)

    return [choice["text"] for choice in response["choices"]]


def batch_answer_same_context(questions: List[str], context: str) -> List[str]:
    """Answers the questions with the given context (remote mode).

    :param questions: The questions to answer.
    :type questions: List[str]
    :param context: The context to answer the questions with.
    :type context: str
    :return: The answers.
    :rtype: List[str]

    """

    parameters = answerer_parameters
    prompts = [
        answerer_prompt.format(question=question, context=context)
        for question in questions
    ]

    response = openai.Completion.create(**parameters, prompt=prompts)

    return [choice["text"] for choice in response["choices"]]


def batch_answer_same_question(question: str, contexts: List[str]) -> List[str]:
    """Answers the question with the given contexts (remote mode).

    :param question: The question to answer.
    :type question: str
    :param contexts: The contexts to answer the question with.
    :type contexts: List[str]
    :return: The answers.
    :rtype: List[str]

    """

    parameters = answerer_parameters
    prompts = [
        answerer_prompt.format(question=question, context=context)
        for context in contexts
    ]

    response = openai.Completion.create(**parameters, prompt=prompts)

    return [choice["text"] for choice in response["choices"]]


def generate(prompt: str, context: str) -> str:
    """Generates a response for the given prompt and context (remote mode).

    :param prompt: The prompt to generate a response for.
    :type prompt: str
    :param context: The context to generate a response for.
    :type context: str
    :return: The generated response.
    :rtype: str

    """

    parameters = generator_parameters
    prompt = generator_prompt.format(prompt=prompt, context=context)

    response = openai.Completion.create(**parameters, prompt=prompt)

    return response["choices"][0]["text"]


def batch_generate(prompts: List[str], contexts: List[str]) -> List[str]:
    """Generates responses for the given prompts and contexts (remote mode).
    
    :param prompts: The prompts to generate responses for.
    :type prompts: List[str]
    :param contexts: The contexts to generate responses for.
    :type contexts: List[str]
    :return: The generated responses.
    :rtype: List[str]

    """

    parameters = generator_parameters
    prompts = [
        generator_prompt.format(prompt=prompt, context=context)
        for prompt, context in zip(prompts, contexts)
    ]

    response = openai.Completion.create(**parameters, prompt=prompts)

    return [choice["text"] for choice in response["choices"]]


def batch_generate_same_context(prompts: List[str], context: str) -> List[str]:
    """Generates responses for the given prompts and context (remote mode).

    :param prompts: The prompts to generate responses for.
    :type prompts: List[str]
    :param context: The context to generate responses for.
    :type context: str
    :return: The generated responses.
    :rtype: List[str]

    """

    parameters = generator_parameters
    prompts = [
        generator_prompt.format(prompt=prompt, context=context) for prompt in prompts
    ]

    response = openai.Completion.create(**parameters, prompt=prompts)

    return [choice["text"] for choice in response["choices"]]


def batch_generate_same_prompt(prompt: str, contexts: List[str]) -> List[str]:
    """Generates responses for the given prompt and contexts (remote mode).

    :param prompt: The prompt to generate responses for.
    :type prompt: str
    :param contexts: The contexts to generate responses for.
    :type contexts: List[str]
    :return: The generated responses.
    :rtype: List[str]

    """

    parameters = generator_parameters
    prompts = [
        generator_prompt.format(prompt=prompt, context=context) for context in contexts
    ]

    response = openai.Completion.create(**parameters, prompt=prompts)

    return [choice["text"] for choice in response["choices"]]
