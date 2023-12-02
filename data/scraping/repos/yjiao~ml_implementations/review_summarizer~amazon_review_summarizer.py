from hyper.contrib import HTTP20Adapter
import requests
from bs4 import BeautifulSoup
import itertools
import re
import logging
import os
from dotenv import load_dotenv
import openai
from typing import Dict, List, Tuple
import sys

# configure a logger to output INFO-level logs from this module only
# this avoids verbose outputs from libraries such as requests
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
logger.addHandler(handler)


def process_review(review_body: str) -> str:
    pattern = re.compile(r" \s+", re.MULTILINE)
    review_body = review_body.strip().replace("\\n", "")
    review_body = review_body.replace("\\", "")
    review_body = pattern.sub("", review_body)
    return review_body


def get_reviews(response_text, include_star_rating: bool = False) -> List[str]:
    soup = BeautifulSoup(response_text, "html.parser")
    all_reviews = []
    for d in soup.find_all("div"):
        div_id = d.attrs.get("id")
        if div_id and div_id.startswith('\\"customer_review'):
            review_body = process_review(
                d.find("span", attrs={"data-hook": '\\"review-body\\"'}).text
            )
            if include_star_rating:
                [star_rating] = d.find("span", class_=['\\"a-icon-alt\\"']).contents
                all_reviews.append(star_rating + ". " + review_body)
            else:
                all_reviews.append(review_body)
    return all_reviews


def get_reviews_by_stars(asin: str) -> Dict[int, List[str]]:
    """Iterate through 1-5 stars and grab the first set of reviews returned by the server."""
    star_to_reviews = dict()
    req_header = {
        ":authority": "www.amazon.com",
        ":method": "POST",
        ":path": "/hz/reviews-render/ajax/reviews/get/ref=cm_cr_getr_d_paging_btm_next_7",
        ":scheme": "https",
        "content-type": "application/x-www-form-urlencoded;charset=UTF-8",  # needed
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36",
    }
    for i, star in enumerate(
        ["one_star", "two_star", "three_star", "four_star", "five_star"], start=1
    ):
        payload = f"sortBy=&reviewerType=all_reviews&formatType=&mediaType=&filterByStar={star}&pageNumber=1&filterByLanguage=&filterByKeyword=&shouldAppend=undefined&deviceType=desktop&canShowIntHeader=undefined&reftag=cm_cr_getr_d_paging_btm_next_7&pageSize=10&asin={asin}&scope=reviewsAjax1"
        sessions = requests.session()
        sessions.mount("https://www.amazon.com", HTTP20Adapter())
        r = sessions.post(
            "https://www.amazon.com/hz/reviews-render/ajax/reviews/get/ref=cm_cr_getr_d_paging_btm_next_7",
            headers=req_header,
            data=payload,
        )

        if r.status_code != 200:
            logger.error(f"Got code {r.status_code} for ASIN {asin}, star {star}.")
            continue
        reviews = get_reviews(r.text)
        star_to_reviews[i] = reviews
    return star_to_reviews


def select_reviews(
    star_order: List[int], stars_to_reviews: Dict[int, List[str]], max_len: int
) -> str:
    """Greedily select reviews that will fit into the prompt.

    A more sophisticated implementation will use Knapsack or other algorithms but
    we are dealing with <10 reviews typically and this is unlikely to make the
    summary significantly better.
    """
    prompt = []
    prompt_len = 0
    for star in star_order:
        for review in stars_to_reviews[star]:
            review_len = len(review)
            if prompt_len + review_len <= max_len:
                prompt.append(review)
                prompt_len += review_len
            else:
                prompt.append(review[-(max_len - prompt_len - review_len) :])
                prompt_len = max_len
                break
    return " ".join(prompt)


def summarize_reviews(asin, max_len=3000 * 4, extra_questions=""):
    """Get reviews from Amazon and call OpenAI API for summarization."""
    logger.info("Retrieving reviews from Amazon...")
    stars_to_reviews = get_reviews_by_stars(asin)

    for star in range(1, 6):
        logger.info(f"# {star}-star reviews retrieved: {len(stars_to_reviews[star])}")

    # allocate same amount of tokens to bad and good reviews
    max_len = (max_len - len(extra_questions)) // 2
    # preferentially use 5 and 1-star reviews
    bad = select_reviews([1, 2, 3], stars_to_reviews, max_len)
    good = select_reviews([5, 4], stars_to_reviews, max_len)

    logger.info(f"Total len of bad reviews: {len(bad)}")
    logger.info(f"Total len of good reviews: {len(good)}")

    context = (
        bad
        + good
        + """
Questions:
1. What are the top 5 pros of this product?
2. What are the top 5 cons of this product?"""
        + extra_questions
        + "\nAnswers:"
    )

    max_tokens = 4000 - len(context) // 4  # use heuristic that 4 char ~ 1 token
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=context,
        temperature=0.9,
        max_tokens=max_tokens,
    )

    summary = response.choices[0].text.strip()
    print("------------------- REVIEW SUMMARY -------------------")
    print(summary)
    print("------------------------------------------------------")
    return summary


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        logger.fatal("OpenAI api key not provided in .env file.")

    print("ASIN of Amazon product: ")
    asin = input()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print("Extra questions (start with 3, <enter> without input to skip): ")
    extra_questions = input()
    print("Starting summarizer...")
    summarize_reviews(asin, extra_questions=extra_questions)


if __name__ == "__main__":
    main()
