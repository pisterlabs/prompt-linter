import time

import openai
import pandas as pd
from joblib import Parallel, delayed
from openai.error import OpenAIError, RateLimitError
from utils import openai_api_key, path


def make_completion(i, line):
    """The make_completion function takes in a line from the posts.csv file and
    uses it to create a completion using OpenAI's API. It then writes that
    completion to the chatgpt_completions folder, which is located in
    data/chatgpt_completions/. The function also prints out how many
    completions have been made so far.

    Parameters
    ----------
        i
            Keep track of the number of posts that have been completed
        line
            Pass the sentence to the make_completion function

    Returns
    -------

        The completion, but we don't need it

    Doc Author
    ----------
        Trelent
    """
    chatgpt_completion = path["data"] / "chatgpt_completions" / f"{i}.txt"
    if not chatgpt_completion.exists():
        prompt = incomplete_prompt.format(sentence=line["sentence"])
        while True:
            try:
                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                response = completion.choices[0].message.content
                break
            except (RateLimitError, OpenAIError) as e:
                if "This model's maximum context length is" in e.user_message:
                    chatgpt_completion.touch()
                    response = ""
                    break
                print(e)
                print("OpenAI error, waiting 10 seconds...")
                time.sleep(10)

        chatgpt_completion.write_text(response)
        print(f"{i + 1}/{len(posts)} | {(i + 1) / len(posts) * 100:.2f}%")
    return


openai.api_key = openai_api_key

incomplete_prompt = (
    "Please write a 250-word text as a psychologist with 20 years of experience in the field of mental "
    "disorders about possible symptoms and diagnoses of mental disorders regarding the user who wrote the post at "
    "the end of this text, which is enclosed in parentheses. Your text should also address whether there are any signs"
    'of ambiguity and/or sarcasm. "{sentence}"'
)

model = "gpt-3.5-turbo"

posts = pd.read_parquet(path["data"] / "full.snappy.parquet")

Parallel(n_jobs=100)(delayed(make_completion)(i, line) for i, line in posts.iterrows())
