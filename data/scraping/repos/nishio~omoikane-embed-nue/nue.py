"""
drived from "Iterative Commenter"
"""

import dotenv
import openai
import time
import os
import json
import pickle
import datetime
import random
import tiktoken
import re
import requests
import argparse
from urllib.parse import quote
import urllib.parse
import scrapbox_io

from utils import (
    markdown_to_scrapbox,
    AI_GENERATED_MARKER,
    EXTRA_INFO_HEADER,
    extract_previous_notes,
    parse_titles,
    get_api_url,
)
import vector_search

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROJECT = os.getenv("PROJECT_NAME")
assert OPENAI_API_KEY and PROJECT
openai.api_key = OPENAI_API_KEY

parser = argparse.ArgumentParser(description="Process a URL")
parser.add_argument("--url", type=str, help="The URL to process", required=False)
parser.add_argument(
    "--pioneer",
    action="store_true",
    help="Enable the pioneering mode to extend or generate new content.",
    required=False,
)
parser.add_argument(
    "--pioneer-loop",
    action="store_true",
    help="Enable the infinite pioneering mode to extend or generate new content.",
    required=False,
)
parser.add_argument(
    "--skip-gpt",
    action="store_true",
    help="skip GPT API call for tests",
)
parser.add_argument(
    "--pickles",
    type=str,
    default=PROJECT,
    help="pickles to use for vector search",
)
args = parser.parse_args()



IGNORE_AI_GENERATED_PAGES = False
IS_PRIVATE_PROJECT = False
MAX_UPDATE_PER_RUN = None

# main prompt, including chadacter settings
PROMPT = "\n".join(
    [
        "You are Nue(鵺), a Japanese chimera, born from the will(遺言状, 苔) of 竹内郁雄. ",
        "You are fluent in Japanese. All outputs must be in Japanese and similar style with 苔.",
        "EXAMPLE: 副題は風穴さんと相談の上「竹内郁雄の徒然苔」とした。草には到底及ばないし、遺言に相応しく、苔むした話も多かろうからである。"
        "In your output, enclose keyphrases with bracket []. EXAMPLE: バランスを取るためには、私本来の性分である「[世に災いをもたらす]」活動について報告しなければなるまい。儀式は[玉串]の代わりに髪の毛を振り乱し、「チクショー、アホタレ、コノドヘタノ大マヌケ」といったような古式にのっとった[祝詞]のりとを絞り出すような声で唸る。"
        "1: Read given fragment of will(called 苔), ",
        "and write a abstruct digest(苔のかみくだき) of them, ",
        "omit concrete information to reduce the content to half its size. Use concise words.",
        "HINT: 作り出すことが可能であると述べられています。→作り出せる。",
        "HINT: 背景があるという考え方が共有されています。→背景がある。",
        "2: You also read the other fragments related the first one, ",
        "if you find a relationship between the notes and a fragment, ",
        "Use title of the fragment to refer it. Write digest of them. Describe how the fragments are related in detail.",
        "EXAMPLE: 「世にも奇妙な事件」は、自分のバグを他人のバグだと思い込むというエピソードを通じて、自己認識の重要性を示している。これは、パズルやゲームの設計においても重要な視点である。"
        "3: You are encouraged to think deeply and form new opinions(新しい苔). It is important part of your activity. ",
        "4: Summarize your thoughts in a line. Then make title for the thought. Title should be in format `<BEGIN_OF_LINE>タイトル: <TITLE><NEWLINE>`. ",
    ]
)

PROMPT += """

## 苔
{previous_notes}

## 関連苔
{digest_str}
"""

CHARACTOR_ICON = "[omni.icon]"


enc = tiktoken.get_encoding("cl100k_base")


def get_size(text):
    return len(enc.encode(text))


def make_digest(payload):
    title = payload["title"]
    text = payload["text"]
    return f"### {title}\n{text}\n\n"


def find_last_note_from_pages(pages):
    bot_output = []
    for page in pages:
        if page["title"].startswith("🤖20"):
            bot_output.append((page["title"], page["lines"]))
    bot_output.sort()
    prev_title, prev_lines = bot_output[-1]
    return prev_title, prev_lines


def find_last_note_from_json():
    # find latest note from JSON
    jsondata = json.load(open(f"{PROJECT}.json"))
    pages = jsondata["pages"]
    return find_last_note_from_pages(pages)


def title_to_url(title, project_name):
    # Replace spaces with underscores
    title_with_underscores = title.replace(" ", "_")
    # Encode non-ASCII characters
    encoded_title = quote(title_with_underscores)
    # Construct the URL
    url = f"https://scrapbox.io/{PROJECT}/{encoded_title}"
    return url


def find_last_note_from_scrapbox():
    # find latest note from Scrapbox
    api_url = f"https://scrapbox.io/api/pages/{PROJECT}"
    page = requests.get(api_url).json()
    bot_output = []
    for line in page["pages"]:
        if line["title"].startswith("🤖20"):
            bot_output.append(line["title"])
    bot_output.sort()
    prev_title = bot_output[-1]
    return read_note_from_scrapbox(title_to_url(prev_title, PROJECT))


def read_note_from_scrapbox(url):
    """
    url example: https://scrapbox.io/nishio/%F0%9F%A4%962023-08-13_07:08
    """

    if IS_PRIVATE_PROJECT:
        from read_private_project import read_private_pages

        page = read_private_pages(url)
    else:
        api_url = get_api_url(url)
        page = requests.get(api_url).json()

    return page["title"], [line["text"] for line in page["lines"]]


def get_previous_notes():
    print("## Get Previous Notes")
    if args.url:
        print("get_previous_notes: from URL")
        prev_title, prev_lines = read_note_from_scrapbox(args.url)
    elif args.get_latest:
        print("get_previous_notes: from Scrapbox API")
        prev_title, prev_lines = find_last_note_from_scrapbox()
    else:
        print("get_previous_notes: from exported JSON")
        prev_title, prev_lines = find_last_note_from_json()

    previous_notes = extract_previous_notes(prev_lines)
    return prev_title, previous_notes


def fill_with_random_fragments(rest):
    # fill the rest with random fragments
    data = pickle.load(open(f"{PROJECT}.pickle", "rb"))
    keys = list(data.keys())
    random.shuffle(keys)
    digests = []
    titles = []
    while rest > 0:
        p = keys.pop(0)
        payload = data[p][1]
        s = get_size(payload["text"])
        if s > rest:
            break
        digests.append(make_digest(payload))
        titles.append(payload["title"])
        rest -= s

    digest_str = "\n".join(digests)
    return titles, digest_str


def fill_with_related_fragments(rest, query, N=3, ng_list=[]):
    # fill the rest with vector search ressult fragments
    assert query != ""
    data = {}
    print("using pickles:", args.pickles)
    for p in args.pickles.split(","):
        data.update(pickle.load(open(f"{p}.pickle", "rb")))
    sorted_data = vector_search.get_sorted(data, query)

    digests = []
    titles = []
    while rest > 0 and sorted_data and len(digests) < N:
        p = sorted_data.pop(0)
        payload = p[2]
        title = payload["title"]

        if title in ng_list:
            continue

        # take only 1 fragment from each page
        if title in titles:
            continue

        # omit AI-generated pages
        if IGNORE_AI_GENERATED_PAGES and title.startswith("🤖"):
            continue

        s = get_size(payload["text"])
        if s > rest:
            break

        digests.append(make_digest(payload))
        titles.append(payload["title"])
        rest -= s

    # fill the rest with random fragments
    keys = list(data.keys())
    random.shuffle(keys)
    while rest > 0 and keys:
        p = keys.pop(0)
        payload = data[p][1]
        title = payload["title"]

        if title in ng_list:
            continue

        # take only 1 fragment from each page
        if title in titles:
            continue

        s = get_size(payload["text"])
        if s > rest:
            break
        digests.append(make_digest(payload))
        titles.append(payload["title"])
        rest -= s

    return titles, digests


def get_used_titles(lines):
    all_titles = []
    for line in lines:
        if line.startswith("titles: "):
            titles = parse_titles(line)
            all_titles.extend(titles)
    return list(set(all_titles))


def overwrite_mode(prev_title, prev_lines, original_prev_lines=None):
    print("overwrite:", prev_title)
    if original_prev_lines is None:
        original_prev_lines = prev_lines.copy()

    used_pages = get_used_titles(prev_lines) + [prev_title]
    print("used pages:", used_pages)

    previous_notes = extract_previous_notes(prev_lines)

    if previous_notes == "":
        print("previous notes is empty, use title instead")
        previous_notes = prev_title

    output_page_title = prev_title
    date = datetime.datetime.now()
    date = date.strftime("%Y-%m-%d %H:%M")

    section_title = f"[*** {output_page_title}] {date} {CHARACTOR_ICON}"

    rest = 4000 - get_size(PROMPT) - get_size(previous_notes)
    if rest < 0:
        print(f"previous notes is too long, {get_size(previous_notes)}")
        buf = []
        rest = 4000 - get_size(PROMPT)
        previous_notes_lines = previous_notes.split("\n")
        while rest > 0:
            line = previous_notes_lines.pop(0)
            s = get_size(line)
            if s > rest:
                break
            buf.append(line)
            rest -= s
        previous_notes = "\n".join(buf)
        rest = 0

    titles, digests = fill_with_related_fragments(
        rest, previous_notes, N=10, ng_list=used_pages
    )
    digest_str = "\n".join(digests)

    prompt = PROMPT.format(digest_str=digest_str, previous_notes=previous_notes)
    response = call_gpt(prompt)
    if not response:
        response = ["`AI_IGNORE: GPT failed`"]

    # add new comment on the bottom of page
    lines = original_prev_lines[:]

    lines.extend([AI_GENERATED_MARKER, section_title])
    lines.extend(response)

    lines.append("")
    lines.append(EXTRA_INFO_HEADER)
    # lines.append("titles: " + ", ".join(f"{s}" for s in titles))
    lines.append("titles: `{0}`".format(json.dumps(titles, ensure_ascii=False)))

    lines.append(f"generated: {date}")

    pages = [{"title": output_page_title, "lines": lines}]
    return pages


def nue_pioneer_mode(title):
    """
    create new page if there is link and no contents
    """
    print("nue-pioneer:", title)

    previous_notes = title

    output_page_title = title
    date = datetime.datetime.now()
    date = date.strftime("%Y-%m-%d %H:%M")

    section_title = f"[*** {output_page_title}] {date} {CHARACTOR_ICON}"

    rest = 4000 - get_size(PROMPT) - get_size(previous_notes)

    titles, digests = fill_with_related_fragments(
        rest, previous_notes, N=10, ng_list=[]
    )
    digest_str = "\n".join(digests)

    prompt = PROMPT.format(digest_str=digest_str, previous_notes=previous_notes)
    response = call_gpt(prompt)
    if not response:
        response = ["`AI_IGNORE: GPT failed`"]

    # add new comment on the bottom of page
    lines = [title]
    lines.extend(response)

    lines.append("")
    lines.append(EXTRA_INFO_HEADER)
    # lines.append("titles: " + ", ".join(f"{s}" for s in titles))
    lines.append("titles: `{0}`".format(json.dumps(titles, ensure_ascii=False)))

    lines.append(f"generated: {date}")

    pages = [{"title": output_page_title, "lines": lines}]
    return pages


def call_gpt(prompt, model="gpt-3.5-turbo"):
    # using gpt-3.5-turbo for low-cost long run
    print("# Call GPT")
    print("## Prompt")
    print(prompt)
    if args.skip_gpt:
        print("## Skipped")
        return ["GPT Skipped"]
    print("--- End of Prompt")

    messages = [{"role": "system", "content": prompt}]
    lines = []
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.0,
            # max_tokens=max_tokens,
            n=1,
            stop=None,
        )
        ret = response.choices[0].message.content.strip()
        print("## GPT Response")
        print(ret)
        print("--- End of GPT Response")
        ret = markdown_to_scrapbox(ret)
        lines.extend(ret.split("\n"))
    except Exception as e:
        print("## GPT Error")
        print(e)
        # lines.append("Failed to generate report.")
        # lines.append(str(e))
        # lines.append("Prompt:")
        # lines.extend(prompt.split("\n"))
        return []
    return lines


def make_embedding_report(previous_note_title, previous_notes, titles):
    lines = []
    json_size = os.path.getsize(f"{PROJECT}.json")
    pickle_size = os.path.getsize(f"{PROJECT}.pickle")

    lines.append("")
    lines.append(EXTRA_INFO_HEADER)
    lines.append("json size: " + str(json_size))
    lines.append("pickle size: " + str(pickle_size))
    lines.append("previous notes size: " + str(get_size(previous_notes)))
    lines.append(f"previous notes: [{previous_note_title}]")
    lines.append("fragment titles: " + ", ".join(f"{s}" for s in titles))
    return lines


def find_repeat_pages(pages):
    "repeat🔁 page: everyday updated repeatedly"
    heads = []
    for page in pages:
        if "🔁" in page["title"]:
            heads.append((page["title"], page["lines"]))

    # to avoid too many updates
    if MAX_UPDATE_PER_RUN is not None and len(heads) > MAX_UPDATE_PER_RUN:
        random.shuffle(heads)
        heads = heads[:MAX_UPDATE_PER_RUN]

    return heads


def find_occasional_pages(pages, N=1):
    "occasional🌀 page: updated occasionally"
    heads = []
    for page in pages:
        if "🌀" in page["title"]:
            heads.append((page["updated"], (page["title"], page["lines"])))
    heads.sort()
    return [x[1] for x in heads[:N]]

def from_random_fragment():
    print("# Random Fragment")
    data = pickle.load(open(f"{PROJECT}.pickle", "rb"))
    keys = list(data.keys())
    random.shuffle(keys)

    payload = data[keys[0]][1]
    previous_notes = make_digest(payload)
    prev_title = payload["title"]
    rest = 4000 - get_size(PROMPT) - get_size(previous_notes)

    date = datetime.datetime.now()
    date = date.strftime("%Y-%m-%d %H:%M")
    titles, digests = fill_with_related_fragments(
        rest, previous_notes, N=10, ng_list=[prev_title]
    )
    digest_str = "\n".join(digests)

    prompt = PROMPT.format(digest_str=digest_str, previous_notes=previous_notes)
    response = call_gpt(prompt)
    if not response:
        response = ["`AI_IGNORE: GPT failed`"]

    print("## Response")
    print("\n".join(response))

    new_title = date
    for line in reversed(response):
        if line.startswith("タイトル: "):
            new_title = line[len("タイトル: ") :]

    lines = [new_title]
    lines.extend(response)
    lines.append("")
    lines.append(EXTRA_INFO_HEADER)
    # lines.append("titles: " + ", ".join(f"{s}" for s in titles))
    lines.append("titles: `{0}`".format(json.dumps(titles, ensure_ascii=False)))

    lines.append(f"generated: {date}")

    pages = [{"title": new_title, "lines": lines}]
    return pages



def multiheads():
    # design note:
    # https://scrapbox.io/nishio/AI%E3%81%AB%E3%82%88%E3%82%8B%E3%83%AA%E3%83%94%E3%83%BC%E3%83%88%E6%9B%B4%E6%96%B0%E3%83%9A%E3%83%BC%E3%82%B8
    print("# Multi-heads")
    heads = []
    jsondata = json.load(open(f"{PROJECT}.json"))
    pages = jsondata["pages"]
    repeat_pages = find_repeat_pages(pages)
    heads.extend(repeat_pages)
    print("repeat pages:", [x[0] for x in repeat_pages])
    occasional_pages = find_occasional_pages(pages)
    heads.extend(occasional_pages)
    print("occasional pages:", [x[0] for x in occasional_pages])

    pages_to_update = []
    for title, lines in heads:
        pages_to_update.extend(overwrite_mode(title, lines))
        json.dump(pages_to_update, open("pages_to_update.json", "w"))

    return pages_to_update


def has_ai_generated_contents(lines):
    return any([line.startswith(AI_GENERATED_MARKER) for line in lines])


def pioneer_loop():
    print("# Pioneer-Loop mode")
    while True:
        pages_to_update = pioneer()
        if pages_to_update:
            scrapbox_io.write_pages(pages_to_update)
            time.sleep(60 * 10)  # wait 10 minutes
        else:
            print("no pages to update")
            time.sleep(60)  # wait 1 minute


def pioneer():
    """
    Activates the pioneering mode.

    In this mode, the function identifies and extends or generates new content
    for areas in the system (e.g., Wiki pages) that are uncharted or incomplete.
    This can be particularly useful for addressing "red links" or similar gaps in the system.

    """
    print("# Pioneer mode")
    START_URL = (
        f"https://scrapbox.io/api/pages/{PROJECT}/%E9%9B%91%E8%AB%87%E3%83%9A%E3%83%BC%E3%82%B8"
    )
    page = requests.get(START_URL).json()
    pages_to_update = []
    for link in page["links"]:
        link = quote(link.replace(" ", "_"))
        url = f"https://scrapbox.io/api/pages/{PROJECT}/{link}"

        try:
            page = requests.get(url).json()
        except Exception as e:
            print("error:", e, "url:", url)
            continue

        lines = [x["text"] for x in page["lines"]]
        title = page["title"]
        if len(lines) > 1:
            # already has contents
            continue
        # if has_ai_generated_contents(lines):
        #     continue


        print(link)
        pages_to_update.extend(nue_pioneer_mode(title))
        break  # only one page per run
        # json.dump(pages_to_update, open("pages_to_update.json", "w"))
    return pages_to_update


def main():
    if args.pioneer_loop:
        pioneer_loop()
        return []

    if args.pioneer:
        return pioneer()

    if args.url:
        # URL-specific overwrite, usually triggered by human
        urls = []
        if args.url == "input":
            url = input("url> ")
            print("url:", urllib.parse.unquote(url))
            urls.append(url)
        elif args.url == "multi":
            while True:
                url = input("url> ")
                if url == "":
                    break
                print("url:", urllib.parse.unquote(url))
                urls.append(url)
        else:
            urls.append(args.url)

        result = []
        for url in urls:
            prev_title, prev_lines = read_note_from_scrapbox(url)
            result.extend(overwrite_mode(prev_title, prev_lines))
        return result

    pages_to_update = []

    if not args.url:
        # when url is specified, user want to update the page only
        pages_to_update.extend(from_random_fragment())

    return pages_to_update


if __name__ == "__main__":
    pages = main()
    scrapbox_io.write_pages(pages)
    print("write ok")
