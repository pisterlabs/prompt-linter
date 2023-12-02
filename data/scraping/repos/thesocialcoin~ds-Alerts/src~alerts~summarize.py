import pandas as pd
from typing import List
import openai
import tiktoken

MAX_TOKENS = 4096


def can_add_tokens(texts: str, new_text: str) -> bool:
    return count_tokens(texts) + count_tokens(new_text) < MAX_TOKENS


def count_tokens(texts: str) -> int:
    encoding = tiktoken.get_encoding("gpt2")
    input_ids = encoding.encode(".\n ".join(texts))
    return len(input_ids)


def select_texts_per_day(data: pd.DataFrame, day: str) -> List[str]:
    # Convert the 'day' string to a datetime object
    day = pd.to_datetime(day)

    # Select rows where the date is 'day' and get the 'text' column
    selected_texts = data.loc[data["date"] == day, "text"]

    # Convert to list and return
    return selected_texts.tolist()


def generate_summarizer(
    texts,
):
    text = f"- {texts[0]}"
    for text_ in texts[1:]:
        if can_add_tokens(text, text):
            text = ".\n- ".join([text, text_])
        else:
            break

    prompt = (
        "Craft a summary using the texts provided, do not use words "
        + f"like news, news roundup, various, opinions or tweets: \n{text}\nTitle: "
    )
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=100,
        temperature=0.7,
        top_p=0.5,
        frequency_penalty=0.5,
        messages=[
            {
                "role": "system",
                "content": "Act as if you are an unbiased journalist and your "
                + "editor tells you to write a summary of texts in english",
            },
            {
                "role": "user",
                "content": f"{prompt}: {text}",
            },
        ],
    )
    return res["choices"][0]["message"]["content"]
