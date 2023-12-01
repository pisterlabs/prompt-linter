# Filename: format_paper.py

import openai


def format_paper(title, abstract, api_key):
    openai.api_key = api_key
    prompt = (
        f"論文の題名と要約を教えてください．タイトル: {title}\n要約: {abstract}"
    )
    response = openai.Completion.create(
        engine="text-davinci-002", prompt=prompt, max_tokens=150, temperature=0.7
    )
    summary = response["choices"][0]["text"].strip()
    return summary
