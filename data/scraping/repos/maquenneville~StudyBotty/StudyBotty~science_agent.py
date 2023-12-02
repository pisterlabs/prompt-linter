# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 02:32:48 2023

@author: marca
"""


from openai_pinecone_tools import *


def science_agent(query, context, model=FAST_CHAT_MODEL):
    # Generate ChatGPT messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "You are my Science Teacher's Assistant.  Provided a science-based query and relevant context, your job is to provide an answer to the query.  Ensure that your answer is rooted in the principles of the most relevant scientific fields, and that the answer is free of speculation",
        },
    ]

    for c in context:
        messages.append({"role": "user", "content": f"Context: {c}"})

    messages.append({"role": "user", "content": f"Query:\n{query}"})

    # Use ChatGPT to generate a Wolfram Alpha natural language query
    answer = generate_response(messages, temperature=0.1, model=model)

    return answer
