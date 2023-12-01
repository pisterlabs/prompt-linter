#! /usr/bin/python

import argparse
import logging
import openai
import time

from bs4 import BeautifulSoup
from bs4.element import Comment
from urllib.request import Request, urlopen

### Constants
DEFAULT_URL = "https://www.yahoo.com"

#### Logger
log = logging.getLogger("root")
log_format = "[%(asctime)s - %(module)25s:%(funcName)-25s] %(message)s"
logging.basicConfig(format=log_format)
# log.setLevel(logging.DEBUG)


def main():
    ### Parse and Display Arguments
    arguments = parse_arguments()

    ### Get Webpage Text
    text = read_url(arguments.url)

    ### Print Open AI Summary
    get_open_ai_summary(arguments.url, text, arguments.key)


def parse_arguments():
    ### Get Arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="URL to parse")
    parser.add_argument("--key", required=True, help="Open AI API Key")
    arguments = parser.parse_args()

    ### Display Arguments
    log.info("URL - %s" % arguments.url)

    return arguments


def read_url(url):
    # From: https://medium.com/@raiyanquaium/how-to-web-scrape-using-beautiful-soup-in-python-without-running-into-http-error-403-554875e5abed
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urlopen(req).read()
    return text_from_html(html)


# From: https://stackoverflow.com/questions/1936466/how-to-scrape-only-visible-webpage-text-with-beautifulsoup
def tag_visible(element):
    if element.parent.name in [
        "style",
        "script",
        "head",
        "title",
        "meta",
        "[document]",
    ]:
        return False
    if isinstance(element, Comment):
        return False
    return True


# From: https://stackoverflow.com/questions/1936466/how-to-scrape-only-visible-webpage-text-with-beautifulsoup
def text_from_html(body):
    soup = BeautifulSoup(body, "html.parser")
    texts = soup.findAll(string=True)
    visible_texts = filter(tag_visible, texts)
    return " ".join(t.strip() for t in visible_texts)


def get_open_ai_summary(url, html, api_key):
    print("URL: " + url + "\n")
    openai.api_key = api_key

    instructions = """You will recieve the content of a webpage, with this content, do the following.
    1) Provide a Title For the Content Received, based on the content of the webpage
    2) Succinctly summarize content received as an abstract.
    3) Extract a list of the top 5 keywords from abstract. Make this a comma separated list.
    4) List 3 Melvil Decimal System ids to tag content received with. Just list the ID and Topic. Make this a comma separated list.

    Example Output would look like this:


Title: The Demon Core and the Strange Death of Louis Slotin

Abstract: This article recounts the story of Louis Slotin, a physicist working on the Manhattan Project, who died as a result of a criticality accident involving a plutonium core. The article explores the details of the accident, the impact of radiation exposure on Slotin's health, and the safety measures implemented afterward.

Keywords: Demon Core, Louis Slotin, criticality accident, plutonium core, radiation exposure

Categories: 539.7 - Atomic and Nuclear Physics, 621.48 - Nuclear Engineering, 940.54 - Military History of World War II

    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "system",
                "content": instructions,
            },
            {
                "role": "user",
                "content": html,
            },
        ],
    )

    response = parse_response(response)

    print(response)


def parse_response(open_ai_response):
    return open_ai_response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    main()
