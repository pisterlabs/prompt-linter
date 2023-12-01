from concurrent.futures import ThreadPoolExecutor
import json
import logging

import openai

from utils import parse_correction_explanations

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger()

PROMPT_TRANSLATE_INPUT = open("prompts/translate_input.txt", "r").read()
PROMPT_CORRECTION_TUPLES = open("prompts/correction_tuples.txt", "r").read()
PROMPT_EXPLAIN_CORRECTION = open("prompts/explain_correction.txt", "r").read()

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


def get_corrected_input(input_str: str) -> str:
    global input_tokens_used, output_tokens_used
    prompt = PROMPT_TRANSLATE_INPUT.format(input_str=input_str)
    logger.info("Making request for corrected input...")
    logger.debug(f"Sending input `{input_str}` for correction")
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    corrected_input = completion.choices[0].message.content.replace('"', "")
    logger.debug(f"Received corrected input `{corrected_input}` for `{input_str}`")
    input_tokens_used += completion.usage.prompt_tokens
    output_tokens_used += completion.usage.completion_tokens
    return corrected_input


def get_correction_tuples(
    input_str: str, corrected_input: str
) -> list[tuple[str, str]]:
    global input_tokens_used, output_tokens_used
    prompt = PROMPT_CORRECTION_TUPLES.format(
        input_text=input_str, corrected_text=corrected_input
    )
    function_definition = {
        "name": "receive_outputs",
        "description": "A function that receives outputs",
        "parameters": {
            "type": "object",
            "properties": {
                "correction_tuples": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                        },
                    },
                    "description": "An array of tuples, where each tuple contains a phrase from the user's input and the corrected version of that phrase",
                },
            },
            "required": [
                "correction_tuples",
            ],
        },
    }
    logger.info("Making request for correction tuples...")
    logger.debug(f"Getting correction tuples for `{input_str}` and `{corrected_input}`")
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": prompt}],
        functions=[function_definition],
        function_call={"name": "receive_outputs"},
        temperature=0.1,
    )
    logger.debug(
        f"Received response for correction tuples for `{input_str}` and `{corrected_input}`"
    )
    resp_dict = json.loads(
        completion.choices[0].message.to_dict()["function_call"]["arguments"]
    )
    correction_tuples = resp_dict["correction_tuples"]
    correction_tuples = [ct for ct in correction_tuples if ct[0] != ct[1]]
    input_tokens_used += completion.usage.prompt_tokens
    output_tokens_used += completion.usage.completion_tokens
    return correction_tuples


def get_correction_explanation(
    input_phrase: str, corrected_phrase: str, entire_correction: str
) -> str:
    global input_tokens_used, output_tokens_used
    prompt = PROMPT_EXPLAIN_CORRECTION.format(
        input_phrase=input_phrase,
        corrected_phrase=corrected_phrase,
        entire_correction=entire_correction,
    )
    logger.info("Making request for correction explanation...")
    logger.debug(
        f"Sending input phrase `{input_phrase}` and corrected phrase `{corrected_phrase}` for correction explanation"
    )
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    correction_explanation = completion.choices[0].message.content
    logger.debug(
        f"Received correction explanation `{correction_explanation}` for input phrase `{input_phrase}` and corrected phrase `{corrected_phrase}`"
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

    with ThreadPoolExecutor() as executor:
        conversation_response_future = executor.submit(
            get_conversation_response, user_input
        )
        corrected_input_future = executor.submit(get_corrected_input, user_input)

        conversation_response = conversation_response_future.result()
        corrected_input = corrected_input_future.result()

        correction_tuples_future = executor.submit(
            get_correction_tuples, user_input, corrected_input
        )
        correction_tuples = correction_tuples_future.result()

        correction_explanations_futures = [
            executor.submit(
                get_correction_explanation,
                input_phrase,
                corrected_phrase,
                corrected_input,
            )
            for input_phrase, corrected_phrase in correction_tuples
        ]
        correction_explanations = [
            future.result() for future in correction_explanations_futures
        ]

    correction_explanation = parse_correction_explanations(
        correction_explanations, validate=False
    )

    correction_response = "{correction}\n\n{explanation}".format(
        correction=corrected_input, explanation=correction_explanation
    )
    return (
        correction_response,
        conversation_response,
        main_message_history,
        input_tokens_used,
        output_tokens_used,
    )
