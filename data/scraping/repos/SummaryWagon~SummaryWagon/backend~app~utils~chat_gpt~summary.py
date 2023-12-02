import openai
from decouple import config
from .count import split_prompt

OPENAI_API_KEY = config("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY


def summarize(text):
    model_engine = "text-davinci-003"
    # 세 문장(영어) 평균 75토큰
    max_tokens = 200

    prompt = f'''Summarize the text below in 3 sentences.\n\n[Start of text]{text}[End of text]'''
    prompt_chunks = split_prompt(prompt, model_engine)

    response = []

    for chunk in prompt_chunks:
        completion = openai.Completion.create(
        engine=model_engine,
        prompt=chunk,
        max_tokens=max_tokens,
        temperature=0.3,
        )
        completions = completion.choices[0].text.strip()
        response.append(completions)

    sentences = " ".join(response).split(". ")
    result = []

    for sentence in sentences:
        if sentence[-1] != ".":
            sentence += "."
        result.append(sentence)

    return result
