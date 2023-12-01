#!/usr/bin/env python

"""social.py

post to social media
"""

__version__ = "0.0.0"

import argparse
import logging
import math
import os
import pathlib
import sys
import textwrap
from http import HTTPStatus

import facebook
import requests
from dotenv import load_dotenv
from langchain import OpenAI, PromptTemplate

from app.core.utilities import DATA_DIR, configure_logging, today_iso_fmt

logger = logging.getLogger(__name__)
load_dotenv()

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
HEALTHCHECKS_FACEBOOK_PING_URL = os.getenv("HEALTHCHECKS_FACEBOOK_PING_URL")

# Facebook
# Note: Better to use a Page Access Token (valid for 2 months)
# References:
# - https://medium.com/nerd-for-tech/automate-facebook-posts-with-python-and-facebook-graph-api-858a03d2b142
FACEBOOK_ACCESS_TOKEN = os.getenv("FACEBOOK_ACCESS_TOKEN")
FACEBOOK_PAGE_ID = os.getenv("FACEBOOK_PAGE_ID")
# - https://developers.facebook.com/docs/facebook-login/guides/access-tokens#usertokens
# - https://stackoverflow.com/questions/18664325/how-to-programmatically-post-to-my-own-wall-on-facebook

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_TOKENS = 4096
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

podcast_transcript = f"{DATA_DIR}/{today_iso_fmt}/{today_iso_fmt}_podcast-content.txt"
podcast_url = f"https://zednews.pages.dev/episode/{today_iso_fmt}/"


def setup():
    configure_logging()

    # cd to the PROJECT_ROOT
    os.chdir(PROJECT_ROOT)


def podcast_is_live(url):
    """Check if the podcast is live"""
    try:
        response = requests.head(url)
        return response.status_code != HTTPStatus.NOT_FOUND
    except requests.exceptions.RequestException:
        return False


def get_content() -> str:
    """Get the content of the podcast transcript"""
    with open(podcast_transcript, "r") as f:
        return f.read()


def create_facebook_post(content: str) -> str:
    """Create a Facebook post using OpenAI's language model."""

    template = """
    Please create a nice facebook post (max 120 words) inviting people to listen to today's podcast whose content is below, highlighting some key news items you consider important, with appropriate usage of bullet points, emojis, whitespace and hashtags.
    Do not add the link to the podcast as it will be added automatically.

    {entry}
    """

    # Calculate the maximum number of tokens available for the prompt
    max_prompt_tokens = MAX_TOKENS - llm.get_num_tokens(template)

    # Trim the content if it exceeds the available tokens
    # TODO: Instead of truncating the content, split it
    # see <https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/split_by_token>
    chars = int(max_prompt_tokens * 3.75)  # Assuming 1 token ≈ 4 chars
    # round down max_chars to the nearest 100
    max_chars = math.floor(chars / 100) * 100
    if len(content) > max_chars:
        content = textwrap.shorten(content, width=max_chars, placeholder=" ...")

    prompt = PromptTemplate(input_variables=["entry"], template=template)

    facebook_post_prompt = prompt.format(entry=content)

    num_tokens = llm.get_num_tokens(facebook_post_prompt)
    logging.info(f"the prompt has {num_tokens} tokens")

    return llm(facebook_post_prompt)


def post_to_facebook(content: str, url: str) -> None:
    """Post a link to the Facebook page"""
    graph = facebook.GraphAPI(access_token=FACEBOOK_ACCESS_TOKEN)
    graph.put_object(
        parent_object=FACEBOOK_PAGE_ID,
        connection_name="feed",
        message=content,
        link=url,
    )
    logger.info(url)
    logger.info(content)


def main(args=None):
    """Console script entry point"""

    if not args:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="social.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.version = __version__
    parser.add_argument("-v", "--version", action="version")

    subparsers = parser.add_subparsers(title="Commands", dest="command")

    # share parser
    share_parser = subparsers.add_parser("share", help="Share to a specific platform")
    share_parser.add_argument(
        "platform",
        choices=["facebook"],
        help="Which platform to share to. Currently only Facebook is supported.",
        type=str,
    )

    # add other parsers here as you please
    # subparsers.add_parser("foo", help="...")

    args = parser.parse_args(args)

    if args.command == "share":
        if args.platform == "facebook" and podcast_is_live(podcast_url):
            try:
                content = get_content()
                facebook_post = create_facebook_post(content)
                post_to_facebook(facebook_post, podcast_url)
                requests.get(HEALTHCHECKS_FACEBOOK_PING_URL)
            except Exception as e:
                logger.error(e)
                requests.get(f"{HEALTHCHECKS_FACEBOOK_PING_URL}/fail")
        else:
            print("Either the podcast is not live or the platform you specified is not supported.")
            sys.exit(1)
    else:
        print("Please specify a valid command.")
        sys.exit(1)


if __name__ == "__main__":
    setup()
    main()
