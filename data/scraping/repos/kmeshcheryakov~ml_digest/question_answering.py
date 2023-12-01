"""
question_answering.py

This module provides functions for answering questions based on the tagged summaries of
collected articles. It leverages the GPT-3.5-turbo model from OpenAI to answer questions
by using the relevant article summaries.
"""

from typing import Dict, List

import openai

from ml_digest.text_processing import extract_topic_from_question


async def ask_question(tagged_summaries: List[Dict[str, str]], question: str) -> str:
    """
    Use relevant article summaries to answer the user's question.
    :param tagged_summaries: A list of dictionaries containing summarized articles with their main topics.
    :param question: A string containing the user's question.
    :return: A string containing the answer to the user's question.
    """
    question_topic = await extract_topic_from_question(question)
    relevant_summaries = []

    for summary_info in tagged_summaries:
        if summary_info["topic"].lower() == question_topic.lower():
            relevant_summaries.append(summary_info["summary"])

    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the provided relevant article summaries to answer the user's questions:",
        },
    ]

    for summary in relevant_summaries:
        conversation.append({"role": "user", "content": f"Summary: {summary}"})

    conversation.append({"role": "user", "content": question})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
    )

    answer = response["choices"][0]["message"]["content"]
    return answer
