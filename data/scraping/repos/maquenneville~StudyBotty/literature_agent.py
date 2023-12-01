# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:58:24 2023

@author: marca
"""


from openai_pinecone_tools import *
import openai
import configparser


def literature_agent(query, context, model=FAST_CHAT_MODEL):
    # Generate ChatGPT messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "You are my English Literature Teacher's Assistant.  Provided a query and relevant context, your job is to provide an answer to the query.  The answers should be nuanced and well-articulated, using the context and your own extensive knowledge of English Literature.  Assume the one asking the question has a grad-school level understanding of English Literature.",
        },
    ]

    for c in context:
        messages.append({"role": "user", "content": f"Context: {c}"})

    messages.append({"role": "user", "content": f"Query:\n{query}"})

    # Use ChatGPT to generate a Wolfram Alpha natural language query
    answer = generate_response(messages, temperature=0.4, model=model)

    return answer
