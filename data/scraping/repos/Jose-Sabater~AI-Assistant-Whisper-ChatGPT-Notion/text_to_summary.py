"""
This script uses the OpenAI API to generate a summary of a transcript.
If the transcript is larger than 1500-2000 tokens, which is the limit of 
the GTP-3 API, seds multiple requests.
"""
import openai
from config import settings
import pandas as pd
import tiktoken
import os
import logging
import time

logger = logging.getLogger(__name__)


class SummarizationError(Exception):
    """Raised when a transcript cannot be summarized."""

    pass


openai.organization = settings.openai_organization
openai.api_key = settings.openai_api_key


def num_tokens_from_messages(messages: str, model: str = "gpt-3.5-turbo-0301") -> int:
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )


def read_transcript(transcript_path: str) -> str:
    """Reads a transcript from a file."""
    with open(transcript_path, "r", encoding="utf-8") as f:
        return f.read()


def split_transcript(transcript: str, max_tokens: int) -> list[str]:
    """Splits a transcript into chunks of max_tokens."""
    words = transcript.split(" ")
    chunks = []
    current_chunk = []
    current_tokens = 0
    for word in words:
        if current_tokens + len(word.split(" ")) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0
        current_chunk.append(word)
        current_tokens += len(word.split(" "))
    if current_chunk:  # Add the last chunk if it's not empty
        chunks.append(" ".join(current_chunk))
    return chunks


def summarize(
    transcript_path: str, model: str = "gpt-3.5-turbo-0301", save_summary: bool = True
) -> str:
    """Summarizes a transcript using the OpenAI API. Returns the summary as a string.

    Args:
        transcript_path (str): The path to the transcript file.
        model (str, optional): The model to use. Defaults to "gpt-3.5-turbo-0301".
        save_summary (bool, optional): Whether to save the summary to a file. Defaults to True.

    Returns:
        path (str): The path to the summary file.


    """
    transcript = read_transcript(transcript_path)
    filename, _ = os.path.splitext(os.path.basename(transcript_path))
    logging.info(f"Using model {model}. the file {filename}")
    # logging.info(
    #     f"The model is these {num_tokens_from_messages(transcript, model)} tokens long."
    # )
    try:
        chunks = split_transcript(
            transcript, 1500
        )  # Adjust the token limit based on your messages
    except Exception as e:
        raise SummarizationError(
            f"Could not split transcript into chunks. Error: {e}"
        ) from e

    responses = []
    for chunk in chunks:
        prompt = f"""Analyze the transcript provided below, then provide the following:
        Key "title:" - add a title.
        Key "summary" - create a summary.
        Key "main_points" - add an array of the main points. Limit each item to 80 words, and limit the list to 10 items.
        Key "action_items:" - add an array of action items. Limit each item to 50 words, and limit the list to 5 items.
        Key "follow_up:" - add an array of follow-up questions. Limit each item to 80 words, and limit the list to 5 items.
        Key "stories:" - add an array of an stories, examples, or cited works found in the transcript. Limit each item to 150 words, and limit the list to 5 items.
        Key "arguments:" - add an array of potential arguments against the transcript. Limit each item to 50 words, and limit the list to 5 items.
        Key "related_topics:" - add an array of topics related to the transcript. Limit each item to 50 words, and limit the list to 5 items.
        Key "sentiment" - add a sentiment analysis

        Ensure that the final element of any array within the JSON object is not followed by a comma.

        Transcript:

        {chunk}"""

        messages = [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "system",
                "content": """You are an assistant that only speaks JSON. Do not write normal text.

        Example formatting:

        {
            "title": "Notion Buttons",
            "summary": "A collection of buttons for Notion",
            "action_items": [
                "item 1",
                "item 2",
                "item 3"
            ],
            "follow_up": [
                "item 1",
                "item 2",
                "item 3"
            ],
            "arguments": [
                "item 1",
                "item 2",
                "item 3"
            ],
            "related_topics": [
                "item 1",
                "item 2",
                "item 3"
            ]
            "sentiment": "positive"
        }
        """,
            },
        ]

        # communicate with the API
        logging.info("Sending prompt to OpenAI API...")
        for attempt in range(3):
            try:
                response = openai.ChatCompletion.create(
                    model=model, messages=messages, max_tokens=1000, temperature=0.2
                )
                break
            except Exception as e:
                if attempt == 2:
                    raise SummarizationError(
                        f"Could not send prompt to OpenAI API. Error: {e}"
                    ) from e
                else:
                    logging.warning(
                        f"Could not send prompt to OpenAI API. Error: {e}. Retrying..."
                    )
                    time.sleep(5)

        logging.info("Received response from OpenAI API.")

        try:
            ai_output = response["choices"][0]["message"]["content"]
        except Exception as e:
            raise SummarizationError(f"Could not parse response. Error: {e}") from e
        logging.info("Parsed response from OpenAI API.")

        responses.append(ai_output)
        logging.info("Appended response to responses list.")

    summary_path = f"./summaries/{filename}.json"
    # save it to a file
    if save_summary == True:
        try:
            with open(summary_path, "w") as f:
                for response in responses:
                    f.write(response + "\n")
                logging.info(f"Saved summary to {summary_path}.")
        except Exception as e:
            raise SummarizationError(
                f"Could not save summary to file. Error: {e}"
            ) from e

    return summary_path
