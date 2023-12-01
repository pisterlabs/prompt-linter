# Pull headlines

import argparse
import os
from pynytimes import NYTAPI
import openai


def retrieve_and_summarize(section="science", length=5):
    nyt = NYTAPI(os.getenv("NYT_API_KEY"), parse_dates=True)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    top_stories = nyt.top_stories(section=section)

    blurb = ""
    for story in top_stories[2:]:
        blurb += (
            story["title"]
            + ". "
            + story["abstract"]
            + " "
            + story["multimedia"][0]["caption"]
        )

    text = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Summarize the following into {length} sentences for a layperson who is a science enthusiast: "
        + blurb,
        max_tokens=50 * int(length),
        temperature=0,
    )

    return text["choices"][0]["text"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="nyt_section_summarizer",
        description="Pick a section: science, politics, sports. Get a 5-sentence summary.",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("-s", "--section", required=False)
    parser.add_argument("-l", "--length", required=False)

    args = parser.parse_args()

    print(retrieve_and_summarize(args.section, args.length))
