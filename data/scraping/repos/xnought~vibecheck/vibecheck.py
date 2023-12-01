from __future__ import annotations


def chatgpt(openai, prompt="Who won the world series in 2020?", model="gpt-3.5-turbo"):
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return resp["choices"][0]["message"]["content"]


def rescale_negative(before: float):
    if before < 0:
        return before * 2
    return before


def parse_vibe(str: str):
    try:
        return float(str)
    except ValueError:
        return 0.0


def gpt_vibecheck(
    personality: str, sentences: list[str], open_ai_key=None, use_env=False
):
    import openai

    if use_env:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    if open_ai_key is not None:
        openai.api_key = open_ai_key
    if not use_env and open_ai_key is None:
        raise ValueError("No openai key provided")

    template = """
    Given you are a person with the personality of "{}"
    Give the sentiment of the following sentence from -1 to 1 
    as a single number where -1 is the most negative and 1 is the most positive

    Sentence: {}
    Sentiment: 
    """

    scores = [
        parse_vibe(chatgpt(openai, template.format(personality, sentence)))
        for sentence in sentences
    ]
    print(scores)
    rescaled_scores = [rescale_negative(score) for score in scores]
    print(rescaled_scores)
    avg = sum(rescaled_scores) / len(rescaled_scores)
    return float(avg)


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()

    sentences = ["I hate you", "I love that I hate you", "I love you"]
    vibe = gpt_vibecheck(
        "I am a computer science student that dislikes unnecessary complexity and is takes peoples suggestions too often",
        sentences,
        use_env=True,
    )
    print(vibe)
