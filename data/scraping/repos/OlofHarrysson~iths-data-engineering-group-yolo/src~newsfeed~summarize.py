# import modules
import argparse
import json
import os
from pathlib import Path

import openai
from dotenv import load_dotenv

from newsfeed.datatypes import BlogSummary
from newsfeed.model import TextSummarizer

# load environment variable from .env file
load_dotenv()

# retrive api key from from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# set api key for making api requests
openai.api_key = OPENAI_API_KEY


def summarize_text(blog_text, non_technical=False, swedish=False):
    # Define a function to summarize text using Openai's API

    if non_technical:
        prompt = f"Summarize the following text concisely with no technical words so that it can be understood by a child: {blog_text}"

    elif swedish == True:
        prompt = f"Summarize the following text concisely in swedish: {blog_text}"

    else:
        prompt = f"Summarize the following text concisely : {blog_text}"

    # create prompt for the api request
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    # extract and return the genereted summary
    summary = response.choices[0].message["content"].strip()
    return summary


def translate_title(title):
    title_prompt = f"Translate this title to swedish: {title}"

    response_trans = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": title_prompt},
        ],
    )

    sv_title = response_trans.choices[0].message["content"].strip()
    return sv_title


def load_articles(blog_name):
    # path of the article with their specified blog name
    articles_path = Path("data/data_warehouse") / blog_name / "articles"

    article_files = [
        file for file in os.listdir(articles_path) if file.endswith(".json")
    ]  # list of all articles

    return article_files


def extract_summaries_from_articles(article_files, blog_name, model_type):
    summaries = []
    if model_type == "local":
        print("Using local model")
        local = TextSummarizer()

    for article_file in article_files:
        summary_file = os.path.join("data/data_warehouse", blog_name, "summaries", article_file)
        print(f"Processing article {article_file}")

        if os.path.isfile(summary_file) == True:
            print(f"\nSummary file {summary_file} already exists\n")

        else:
            with open(Path("data/data_warehouse", blog_name, "articles", article_file), "r") as f:
                article_data = json.load(f)

            blog_text = article_data["blog_text"]
            article_title = article_data["title"]

            if model_type == "local":
                summary = local.summerize_text_local(blog_text, non_technical=False, swedish=False)
                simple_summary = local.summerize_text_local(
                    blog_text, non_technical=True, swedish=False
                )
                swedish_summary = local.summerize_text_local(
                    blog_text, non_technical=False, swedish=True
                )

            if model_type == "gpt":
                summary = summarize_text(blog_text, non_technical=False, swedish=False)
                simple_summary = summarize_text(blog_text, non_technical=True, swedish=False)
                swedish_summary = summarize_text(blog_text, non_technical=False, swedish=True)
                title_translator = translate_title(article_title)

            unique_id = article_data["unique_id"]
            link = article_data["link"]
            published = article_data["published"]

            blog_summary = BlogSummary(
                unique_id=unique_id,
                title=article_title,
                text=summary,
                simple=simple_summary,
                swedish=swedish_summary,
                swe_title=title_translator,
                link=link,
                published=published,
            )
            print(blog_summary)
            summaries.append(blog_summary)

            # saves summary
            save_dir = Path("data/data_warehouse", blog_name, "summaries")
            save_dir.mkdir(exist_ok=True, parents=True)
            save_path = save_dir / blog_summary.get_filename()
            with open(save_path, "w") as f:
                f.write(blog_summary.json(indent=2))

    return summaries


def main(blog_name, model_type):
    print(f"Processing {blog_name}")
    article_files = load_articles(blog_name)
    summaries = extract_summaries_from_articles(article_files, blog_name, model_type)
    print(f"Done processing {blog_name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blog_name", type=str, default="mit", choices=["mit", "big_data"])
    parser.add_argument("--model_type", type=str, default="gpt", choices=["gpt", "local"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(blog_name=args.blog_name, model_type=args.model_type)
