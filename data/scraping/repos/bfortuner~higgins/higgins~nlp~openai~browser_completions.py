import os
import time
from typing import Any

import openai

from higgins.nlp import nlp_utils

from . import caching
from . import completion_utils


openai.api_key = os.getenv("OPENAI_API_KEY")


def open_website_completion(cmd: str, engine="davinci", cache: Any = None):
    prompt = f"""Convert the following text into commands:

    Q: goto amazon.com
    A: `OpenWebsite` PARAMS website=>amazon.com <<END>>
    Q: open Target website
    A: `OpenWebsite` PARAMS website=>target.com <<END>>
    Q: leetcode.com
    A: `OpenWebsite` PARAMS website=>leetcode.com <<END>>
    Q: goto Ebay
    A: `OpenWebsite` PARAMS website=>ebay.com <<END>>
    Q: Open the openai website
    A: `OpenWebsite` PARAMS website=>open.ai <<END>>
    Q: go to facebook homepage
    A: `OpenWebsite` PARAMS website=>facebook.com <<END>>
    Q: go to website
    A: `OpenWebsite` PARAMS website=>??? <<END>>
    Q: Open wikipedia
    A: `OpenWebsite` PARAMS website=>wikipedia.org <<END>>
    Q: Open the New York Times website
    A: `OpenWebsite` PARAMS website=>nyt.com <<END>>
    Q: {cmd}
    A:"""
    cache = cache if cache is not None else caching.get_default_cache()
    cache_key = nlp_utils.hash_normalized_text(prompt)
    if cache_key not in cache:
        start = time.time()
        response = openai.Completion.create(
            engine=engine,
            model=None,
            prompt=prompt,
            temperature=0.2,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.2,
            presence_penalty=0.0,
            stop=["<<END>>"],
        )
        # print(f"Time: {time.time() - start:.2f}")
        answer = response["choices"][0]["text"].strip("Q:").strip()
        cache.add(
            key=cache_key,
            value={
                "cmd": cmd,
                "answer": answer,
                "response": response
            }
        )
    else:
        answer = cache[cache_key]["answer"]
        response = cache[cache_key]["response"]

    return answer


def web_navigation_completion(cmd: str, engine="davinci", cache: Any = None):
    prompt = f"""Convert the following text into commands:

    Q: Go to my amazon cart
    A: `OpenWebsite` PARAMS website=>www.amazon.com -> `ClickLink` PARAMS link_text=>cart <<END>>
    Q: open my github pull requests
    A: `OpenWebsite` PARAMS website=>www.github.com -> `ClickLink` PARAMS link_text=>pull requests <<END>>
    Q: search wikipedia for grizzly bears
    A: `OpenWebsite` PARAMS website=>www.wikipedia.org -> `SearchOnWebsite` PARAMS text=>grizzly bears ### filter=>??? <<END>>
    Q: search amazon for ski mask filter for Prime only
    A: `OpenWebsite` PARAMS website=>www.amazon.com -> `SearchOnWebsite` PARAMS text=>ski mask ### filter=>Prime <<END>>
    Q: go to openai homepage
    A: `OpenWebsite` PARAMS website=>www.open.ai <<END>>
    Q: leetcode.com
    A: `OpenWebsite` PARAMS website=>leetcode.com <<END>>
    Q: search twitter for $index mentions
    A: `OpenWebsite` PARAMS website=>www.twitter.com -> `SearchOnWebsite` PARAMS text=>$index ### filter=>??? <<END>>
    Q: Sign out of my account
    A: `SignOutOfWebsite` PARAMS website=>??? <<END>>
    Q: Sign out of my Amazon account
    A: `SignOutOfWebsite` PARAMS website=>www.amazon.com <<END>>
    Q: Login to my new york times account
    A: `OpenWebsite` PARAMS website=>www.nyt.com -> `LogInToWebsite` PARAMS website=>??? ### username=>??? ### password=>??? <<END>>
    Q: search for hard-shell rain jackets on ebay
    A: `OpenWebsite` PARAMS website=>www.ebay.com -> `SearchOnWebsite` PARAMS text=>hard-shell rain jackets ### filter=>??? <<END>>
    Q: open walmart
    A: `OpenWebsite` PARAMS website=>www.walmart.com <<END>>
    Q: search wikipedia
    A: `OpenWebsite` PARAMS website=>www.wikipedia.org -> `SearchOnWebsite` PARAMS text=>??? ### filter=>??? <<END>>
    Q: log out
    A: `SignOutOfWebsite` PARAMS website=>??? <<END>>
    Q: log in
    A: `LogInToWebsite` PARAMS website=>??? ### username=>??? password=>???<<END>>
    Q: open facebook marketplace
    A: `OpenWebsite` PARAMS website=>www.facebook.com -> `ClickLink` PARAMS link_text=>marketplace <<END>>
    Q: Go to circle ci and login with the username bfortuner
    A: `LogInToWebsite` PARAMS website=>www.circleci.com ### username=>bfortuner ### password=>??? <<END>>
    Q: {cmd}
    A:"""
    cache = cache if cache is not None else caching.get_default_cache()
    cache_key = nlp_utils.hash_normalized_text(prompt)
    if cache_key not in cache:
        start = time.time()
        response = openai.Completion.create(
            engine=engine,
            model=None,
            prompt=prompt,
            temperature=0.2,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.2,
            presence_penalty=0.0,
            stop=["<<END>>"],
        )
        # print(f"Time: {time.time() - start:.2f}")
        answer = response["choices"][0]["text"].strip("Q:").strip()
        cache.add(
            key=cache_key,
            value={
                "cmd": cmd,
                "answer": answer,
                "response": response
            }
        )
    else:
        answer = cache[cache_key]["answer"]
        response = cache[cache_key]["response"]

    return answer


if __name__ == "__main__":
    examples = [
        "sign in to my yahoo account",
        "go to target.com",
        "find me airpods on ebay",
        "search wikipedia",
        "search google",
        "search bing for Harley-Davidson motorcycles",
    ]
    for text in examples:
        answer = web_navigation_completion(text)
        intent = completion_utils.convert_string_to_action_chain(answer)
        print(f"Q: {text}\nA: {answer}\nI: {intent}")
