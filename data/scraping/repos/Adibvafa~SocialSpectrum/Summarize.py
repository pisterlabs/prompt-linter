"""
Summarize.py
-------------
Summarizes the created course
"""

import openai


def Create_Summary(paragraphs):
    passage = ''.join(paragraphs)

    prompt = "Summarize and shorten as much as possible while keeping the important points and essence of the passage: " + passage
    short_paragraph = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=250,
        top_p=1)

    return short_paragraph["choices"][0]["message"]["content"].strip()
