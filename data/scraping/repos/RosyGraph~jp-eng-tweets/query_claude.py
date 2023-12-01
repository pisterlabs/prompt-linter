import json
from pathlib import Path as P

import dotenv
import pandas as pd
from anthropic import AI_PROMPT, Anthropic, HUMAN_PROMPT

dotenv.load_dotenv()

SEED = 42
NUM_TWEETS = 50

JP_PROMPT = "Produce an explanation in Japanese of following Tweet from Twitter user @{author}.\n\n{content}"
ENG_PROMPT = "Produce an explanation of following Tweet from Twitter user @{author}.\n\n{content}"
REPORTS_DIR = P(__file__).parent.parent / "reports"
OUTPUT_FILE = REPORTS_DIR / "tweets_report.json"

tweets_df = pd.read_csv("data/tweets.csv")
sample_tweets = tweets_df.sample(n=NUM_TWEETS, random_state=SEED)[["author", "content"]]

anthropic = Anthropic()


def query_claude(tweet_row, prompt):
    author = tweet_row["author"]
    content = tweet_row["content"]
    prompt = f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}".format(
        author=author, content=content
    )

    return anthropic.completions.create(
        model="claude-instant-1.2", max_tokens_to_sample=300, prompt=prompt
    ).completion


output_list = []

for _, row in sample_tweets.iterrows():
    jp_explanation = query_claude(row, JP_PROMPT)
    eng_explanation = query_claude(row, ENG_PROMPT)

    output_list.append(
        {
            "author": row["author"],
            "content": row["content"],
            "jp_explanation": jp_explanation,
            "en_explanation": eng_explanation,
        }
    )

with open(OUTPUT_FILE, "w") as f:
    json.dump(output_list, f, indent=2, ensure_ascii=False)
