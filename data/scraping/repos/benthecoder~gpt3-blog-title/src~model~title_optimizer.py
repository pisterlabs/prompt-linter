import argparse
import os
import re
from math import exp

import openai
import pandas as pd
from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
assert os.getenv("OPENAI_API_KEY"), "No OPENAI_API_KEY defined in .env."

# set api_key
openai.api_key = os.environ["OPENAI_API_KEY"]


def get_pred(res):
    """Extract predicted class and probability from the finetune model response"""

    pred = res["choices"][0]["text"]
    class_prob = exp(res["choices"][0]["logprobs"]["token_logprobs"][0])

    if pred == " bad":
        class_prob = 1.0 - class_prob

    return pred, class_prob


def title_optimizer(fine_tuned_model, title_input, temp=1.0):

    base_prompt = (
        "Rewrite the following blog post title into 6 different titles but optimized for virality"
        " and quality: {0}\n\n-"
    )
    finetune_prompt = "Title: {0} ->"

    r = openai.Completion.create(
        model="text-davinci-003",
        prompt=base_prompt.format(title_input),
        temperature=temp,  # 0 is deterministic output; should set to 0.7 or 1 elsewise
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    gen_titles = re.split(r" ?\n-", r["choices"][0]["text"])

    gen_titles = list(set(gen_titles + [title_input]))

    ranked_titles = []

    for gen_title in gen_titles:

        # remove numbered prefix
        gen_title = re.sub(r"^\d+\. ", "", gen_title)

        r = openai.Completion.create(
            model=fine_tuned_model,
            prompt=finetune_prompt.format(gen_title),
            temperature=0,
            max_tokens=1,
            logprobs=1,
        )

        _, class_prob = get_pred(r)

        ranked_titles.append((gen_title, class_prob))

    df = pd.DataFrame(ranked_titles, columns=["Title", "Good Prob"])
    df = df.sort_values(by="Good Prob", ascending=False)

    # highlight the input text and format percentages
    df_styled = (
        df.style.hide(axis="index")
        .applymap(
            lambda x: "color: #00FF00" if x == title_input else "",
            subset=["Title"],
        )
        .format({"Good Prob": "{:.2%}"})
    )

    return df, df_styled


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fine_tuned_model",
        type=str,
        help="Fine-tuned model name to use for title optimization",
    )
    parser.add_argument(
        "--title_input",
        type=str,
        help="Blog title to optimize",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="Sampling temperature to use for GPT-3 completion",
    )

    args = parser.parse_args()

    df, _ = title_optimizer(args.fine_tuned_model, args.title_input, args.temp)

    # display the titles and scores in the terminal without truncating the titles
    with pd.option_context("display.max_colwidth", None):
        print(df.to_string(index=False))
