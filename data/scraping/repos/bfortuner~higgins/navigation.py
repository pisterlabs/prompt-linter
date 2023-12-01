import json
from pathlib import Path
import os
import time

import openai

from . import OPENAI_CACHE_DIR


openai.api_key = os.getenv("OPENAI_API_KEY")

WEB_NAV_CACHE_FILE = os.path.join(OPENAI_CACHE_DIR, "web_nav.json")

WEB_NAVIGATION_PROMPT = """Q: Go to my amazon cart
A: ChangeURL `amazon.com` -> ClickLink `cart`
Q: open my github pull requests
A: ChangeURL `http://www.github.com` -> ClickLink `pull requests`
Q: search google for tents
A: ChangeURL `google.com` -> FindSearchBar -> TypeText `tents` -> PressKey `enter`
Q: search amazon for ski mask
A: ChangeURL `amazon.com` -> FindSearchBar -> TypeText `ski mask` -> PressKey `enter`
Q: open facebook marketplace
A: ChangeURL `www.facebook.com` -> ClickLink `marketplace`
Q: go to openai homepage
A: ChangeURL `open.ai`
Q: search twitter for $index mentions
A: ChangeURL `twitter.com` -> FindSearchBar -> TypeText `$index` -> PressKey `enter`
Q: open my youtube profile page
A: ChangeURL `www.youtube.com` -> ClickLink `profile`
Q: search bestbuy for smart tv
A: ChangeURL `www.bestbuy.com` -> FindSearchBar -> TypeText `smart tv` -> PressKey `enter`
Q: search for roger federer highlights on youtube
A: ChangeURL `www.youtube.com` -> FindSearchBar -> TypeText `roger federer` -> PressKey `enter`
Q: search twitter for latest #elonmusk tweets
A: ChangeURL `twitter.com` -> FindSearchBar -> TypeText `#elonmusk` -> PressKey `enter`
Q: search for backpacks at REI
A: ChangeURL `www.rei.com` -> FindSearchBar -> TypeText `backpacks` -> PressKey `enter`
Q: Login to my amazon account
A: ChangeURL `amazon.com` -> ClickLink `sign in`
Q: Sign out of my account
A: ClickLink `sign out`
Q: Logout
A: ClickLink `logout`
Q: Login to my new york times account
A: ChangeURL `www.nytimes.com` -> ClickLink `sign in`
Q: {question}
A:"""
WEB_NAVIGATION_PROMPT = """Q: Go to my amazon cart
A: ChangeURL `amazon.com` -> ClickLink `cart`
Q: open my github pull requests
A: ChangeURL `http://www.github.com` -> ClickLink `pull requests`
Q: search google for tents
A: ChangeURL `google.com` -> FindSearchBar -> TypeText `tents` -> PressKey `enter`
Q: search amazon for ski mask
A: ChangeURL `amazon.com` -> FindSearchBar -> TypeText `ski mask` -> PressKey `enter`
Q: open facebook marketplace
A: ChangeURL `www.facebook.com` -> ClickLink `marketplace`
Q: go to openai homepage
A: ChangeURL `open.ai`
Q: search twitter for $index mentions
A: ChangeURL `twitter.com` -> FindSearchBar -> TypeText `$index` -> PressKey `enter`
Q: Sign out of my account
A: ClickLink `sign out`
Q: Login to my new york times account
A: ChangeURL `www.nytimes.com` -> ClickLink `sign in`
Q: search for hard-shell rain jackets on ebay
A: ChangeURL `ebay.com` -> FindSearchBar -> TypeText `hard-shell rain jackets` -> PressKey `enter`
Q: open yahoo
A: ChangeURL `yahoo.com`
Q: {question}
A:"""
start_sequence = "\nA:"
restart_sequence = "\nQ:"


OPENAI_ENGINES = [
    "davinci",
    "curie",
    "davinci-instruct-beta",
]

FINED_TUNED_MODELS = [
    # fine-tuned on 11k bash commands from nl2bash (4 epochs, lr_multipler .1, bs 4)
    "curie:ft-user-7rs1dte2m2824vd5bddi84s8-2021-07-29-21-05-24",
    # fine-tuned on 11k bash commands from nl2bash (1 epoch1, lr_multipler .05, bs 8)
    "curie:ft-user-7rs1dte2m2824vd5bddi84s8-2021-07-29-23-29-05",
]


def ask_web_navigation_model(
    cmd: str, engine="davinci", cache_path: str = WEB_NAV_CACHE_FILE
):
    # Check cache to avoid API calls
    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path))

    if cmd not in cache:
        start = time.time()
        prompt = WEB_NAVIGATION_PROMPT.format(question=cmd)
        print(prompt)
        response = openai.Completion.create(
            engine=engine,
            model=None,
            prompt=prompt,
            temperature=0.2,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.2,
            presence_penalty=0.0,
            stop=["\n"],
        )
        print(f"Time: {time.time() - start:.2f}")
        answer = response["choices"][0]["text"]
        cache[cmd] = {
            "cmd": cmd,
            "answer": answer,
            "response": response
        }
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(cache, open(cache_path, "w"))
    else:
        print(f"Cache Hit. Loading {cmd} from cache")
        answer = cache[cmd]["answer"]
        response = cache[cmd]["response"]

    answer = answer.strip("Q:").strip()
    return answer
