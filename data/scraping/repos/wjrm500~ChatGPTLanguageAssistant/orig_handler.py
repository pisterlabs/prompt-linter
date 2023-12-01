from concurrent.futures import ThreadPoolExecutor
import logging
import re

import openai

from utils import parse_correction_explanations

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger()

PROMPT_ANALYSE_CORRECTION = open("prompts/analyse_correction.txt", "r").read()
PROMPT_TRANSLATE_SENTENCE = open("prompts/translate_sentence.txt", "r").read()

main_message_history: list | None = None
input_tokens_used: int | None = None
output_tokens_used: int | None = None


def get_conversation_response(user_input: str) -> str:
    global main_message_history, input_tokens_used, output_tokens_used
    if main_message_history is None:
        raise ValueError("main_message_history is not set")
    main_message_history.append({"role": "user", "content": user_input})
    logger.info("Making request for conversation response...")
    logger.debug(f"Sending user input `{user_input}` for conversation response")
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=main_message_history, temperature=0.8
    )
    conversation_response = completion.choices[0].message.content
    logger.debug(
        f"Received conversation response `{conversation_response}` for `{user_input}`"
    )
    main_message_history.append({"role": "assistant", "content": conversation_response})
    input_tokens_used += completion.usage.prompt_tokens
    output_tokens_used += completion.usage.completion_tokens
    return conversation_response


def get_corrected_sentence(input_sentence: str) -> str:
    global input_tokens_used, output_tokens_used
    prompt = PROMPT_TRANSLATE_SENTENCE.format(sentence=input_sentence)
    logger.info("Making request for corrected sentence...")
    logger.debug(f"Sending input sentence `{input_sentence}` for correction")
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    corrected_sentence = completion.choices[0].message.content.replace('"', "")
    logger.debug(
        f"Received corrected sentence `{corrected_sentence}` for `{input_sentence}`"
    )
    input_tokens_used += completion.usage.prompt_tokens
    output_tokens_used += completion.usage.completion_tokens
    return corrected_sentence


def get_correction_explanation(input_sentence: str, corrected_sentence: str) -> str:
    global input_tokens_used, output_tokens_used
    prompt = PROMPT_ANALYSE_CORRECTION.format(
        input_sentence=input_sentence, corrected_sentence=corrected_sentence
    )
    logger.info("Making request for correction explanation...")
    logger.debug(
        f"Sending input sentence `{input_sentence}` and corrected sentence `{corrected_sentence}` for correction explanation"
    )
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    correction_explanation = completion.choices[0].message.content
    logger.debug(
        f"Received correction explanation `{correction_explanation}` for input sentence `{input_sentence}` and corrected sentence `{corrected_sentence}`"
    )
    input_tokens_used += completion.usage.prompt_tokens
    output_tokens_used += completion.usage.completion_tokens
    return correction_explanation


def call_api(
    user_input: str,
    _main_message_history: list,
    _input_tokens_used: int,
    _output_tokens_used: int,
) -> tuple:
    global main_message_history, input_tokens_used, output_tokens_used
    main_message_history = _main_message_history
    input_tokens_used = _input_tokens_used
    output_tokens_used = _output_tokens_used
    logger.info("Chat initiated by user...")

    split_regex = r"(?<=[.!?])\s+"
    input_sentences = re.split(split_regex, user_input)

    with ThreadPoolExecutor() as executor:
        conversation_response_future = executor.submit(
            get_conversation_response, user_input
        )
        corrected_sentences_futures = [
            executor.submit(get_corrected_sentence, sentence)
            for sentence in input_sentences
        ]

        conversation_response = conversation_response_future.result()
        corrected_sentences = [
            future.result() for future in corrected_sentences_futures
        ]

        correction_explanations_futures = [
            executor.submit(
                get_correction_explanation, input_sentence, corrected_sentence
            )
            for input_sentence, corrected_sentence in zip(
                input_sentences, corrected_sentences
            )
        ]
        correction_explanations = [
            future.result() for future in correction_explanations_futures
        ]

    correction_explanation = parse_correction_explanations(correction_explanations)

    correction_response = "{correction}\n\n{explanation}".format(
        correction=" ".join(corrected_sentences), explanation=correction_explanation
    )
    return (
        correction_response,
        conversation_response,
        main_message_history,
        input_tokens_used,
        output_tokens_used,
    )
