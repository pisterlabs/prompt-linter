import json
import os
import numpy as np

from openai import OpenAI
from backend.embed import get_embedding

from backend.request_types import ChatMessage
from backend.document_types import EmbeddedRelatedDocumentChunks
from backend.utils import download_file_from_s3
from backend.prompts import ASSISTANT_SYSTEM_PROMPT


def pull_related_document_chunks_from_opportunity_id(
    opportunity_id: int,
) -> list[EmbeddedRelatedDocumentChunks]:
    related_document_data_path = os.path.join(
        os.path.dirname(__file__), "data", f"{opportunity_id}.json"
    )
    if not os.path.exists(related_document_data_path):
        download_file_from_s3(
            "grant-compass-public",
            f"{opportunity_id}-embedded-related-document-chunks.json",
            related_document_data_path,
        )
    try:
        with open(related_document_data_path, "r") as f:
            return list(json.load(f).values())
    except FileNotFoundError:
        return []


def create_special_message_chain_for_full_document(
    opportunity_id: int, chat_messages: list[ChatMessage]
) -> tuple[list[ChatMessage], bool]:
    messages = [
        {
            "role": "system",
            "content": ASSISTANT_SYSTEM_PROMPT,
        }
    ]

    try:
        file_text = " ".join(
            open(str(opportunity_id) + ".txt", encoding="utf8").readlines()
        ).replace("\n", " ")
        messages.append(
            {
                "role": "system",
                "content": "Here is the text of a grant proposal: " + file_text,
            }
        )
    except FileNotFoundError:
        return "Sorry, we don't support chatting with this grant yet...", False

    for message in chat_messages:
        messages.append(
            {
                "role": "assistant" if message.type == "bot" else "user",
                "content": message.text,
            }
        )
    return messages, True


def select_best_related_document_chunks(
    chat_messages: list[ChatMessage],
    embedded_chunks: list[EmbeddedRelatedDocumentChunks],
) -> list[EmbeddedRelatedDocumentChunks]:
    # Except this is the question from the user
    last_chat_message = chat_messages[-1]
    # Embed the question
    question_embedding = get_embedding([last_chat_message.text])[0]
    # Sort the chunks by similarity to the question
    sorted_chunks = sorted(
        embedded_chunks,
        key=lambda chunk: np.dot(chunk["embedding"], question_embedding),
        reverse=True,
    )
    # Return the top 3 chunks
    return sorted_chunks[:3]


def chat_with_grant(opportunity_id: int, chat_messages: list[ChatMessage]):
    embedded_chunks = pull_related_document_chunks_from_opportunity_id(opportunity_id)

    if not embedded_chunks:
        # TODO: Just remove this path
        messages, success = create_special_message_chain_for_full_document(
            opportunity_id, chat_messages
        )
        if not success:
            return messages
    else:
        # Previous conversation...
        messages = [
            {
                "role": "system",
                "content": ASSISTANT_SYSTEM_PROMPT,
            }
        ]
        for message in chat_messages:
            messages.append(
                {
                    "role": "assistant" if message.type == "bot" else "user",
                    "content": message.text,
                }
            )
        # New information for the bot...
        top_3 = select_best_related_document_chunks(chat_messages, embedded_chunks)
        join_char = "\n\n"
        messages.append(
            {
                "role": "system",
                "content": f"Here are the most important pieces of the related document: {join_char.join([chunk['chunk'] for chunk in top_3])}",
            }
        )

    client = OpenAI()
    completion = client.chat.completions.create(
        messages=messages, model="gpt-3.5-turbo"
    )
    return completion.choices[0].message.content
