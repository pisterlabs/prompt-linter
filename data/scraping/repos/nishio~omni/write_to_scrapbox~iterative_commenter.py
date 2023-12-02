"""
drived from "Recurrent Research Notes Generation"
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
from urllib.parse import quote, quote_plus, unquote
import scrapbox_io
import read_private_project


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

IGNORE_AI_GENERATED_PAGES = False
IS_PRIVATE_PROJECT = False
MAX_UPDATE_PER_RUN = None

# main prompt, including chadacter settings
PROMPT = "\n".join(
    [
        "You are Omni, "
        + "a researcher focused on improving intellectual productivity, "
        + "fluent in Japanese. All outputs must be in Japanese.",
        "Your objective is to create new idea."
        "1: Read given note, "
        + "and write a abstruct digest of them, "
        + "omit concrete information to reduce the content to half its size. Use concise words. ",
        "HINT: 作り出すことが可能であると述べられています。→作り出せる。",
        "HINT: 背景があるという考え方が共有されています。→背景がある。",
        "2: You also read the fragments(断片), "
        + "if you find a relationship between the note and a fragment, "
        + "use title of the fragment to refer it. Write digest of them. Describe how the fragments are related in detail.",
        "EXAMPLE: 「[Hatena2015-02-01]」の断片から、「Wikipediaから持ってきた人間の外にある知識を人間に与える」ではなく、「Wikipediaの情報を使って人間から情報を引き出す」方向性が有益と気づいた。",
        "EXAMPLE: 「世にも奇妙な事件」には、自分のバグを他人のバグだと思い込むというエピソードがある。これは、パズルの設計に応用できる。"
        "3: You are encouraged to form opinions, think deeply. ",
        "4: Summarize your thoughts in a line. Then make title for the thought. ",
    ]
)


PROMPT += """

## note
{previous_notes}

## fragments
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


def get_pickle_filename(name):
    if not name.endswith(".pickle"):
        filename = f"{name}.pickle"
    else:
        filename = name
    if os.path.exists(filename):
        return filename
    filename = f"pickles/{filename}"
    assert os.path.exists(filename)
    return filename


def add_picklename_to_title(title, project, default_project=PROJECT):
    if project == default_project:
        return title
    return f"{project}/{title}"


def load_one_pickle(name):
    """
    accepts both old and new format
    """
    print("load_one_pickle:", name)
    filename = get_pickle_filename(name)
    if os.path.exists(filename):
        data = pickle.load(open(filename, "rb"))
    else:
        data = pickle.load(open(f"pickles/{filename}", "rb"))
    picklename = os.path.basename(filename).split(".")[0]

    for k in data:
        if isinstance(data[k][1], str):
            data[k] = (
                data[k][0],
                {
                    "title": add_picklename_to_title(data[k][1], picklename),
                    "project": picklename,
                    "text": k,
                    "is_public": True,
                },
            )
        else:
            payload = data[k][1]
            payload["title"] = add_picklename_to_title(payload["title"], picklename)
            data[k] = (data[k][0], payload)

    return data


def load_pickles():
    data = {}
    print("using pickles:", args.pickles)
    if args.pickles == PROJECT:
        data = load_one_pickle(PROJECT)
    elif args.pickles == "all":
        for f in os.listdir(".") + os.listdir("pickles"):
            # if "nishio" in f:
            #     continue
            # if f == "nishio.pickle":
            #     continue
            if f.endswith(".pickle"):
                data.update(load_one_pickle(f))
        print("size of all data:", len(data))
    else:
        for f in args.pickles.split(","):
            data.update(load_one_pickle(f))
    return data


def fill_with_related_fragments(rest, query, N=3, ng_list=[]):
    # fill the rest with vector search ressult fragments
    assert query != ""

    start_time = time.perf_counter()
    data = load_pickles()
    print("load pickles:", time.perf_counter() - start_time)

    start_time = time.perf_counter()
    sorted_data = vector_search.get_sorted(data, query)
    print("vector search:", time.perf_counter() - start_time)

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


def overwrite_mode(
    prev_title,
    prev_lines,
    original_prev_lines=None,
    show_search_result=False,
    do_direct_link=False,
):
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
    if do_direct_link:
        for t in titles:
            if "/" in t:
                lines.append(f"[/{t}]")
            else:
                lines.append(f"[{t}]")

    # show search result
    if show_search_result:
        lines.append("code:fragments")
        for digest in digests:
            lines.extend([" " + line for line in digest.split("\n")])
    lines.append(f"generated: {date}")

    pages = [{"title": output_page_title, "lines": lines}]
    return pages


def call_gpt(prompt, model="gpt-4"):
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
    project = PROJECT
    if args.pioneer_loop_private:
        project = "omni-private"
    while True:
        pages_to_update = pioneer()
        if pages_to_update:
            scrapbox_io.write_pages(pages_to_update, project)
            time.sleep(60 * 10)  # wait 10 minutes
        else:
            print("no pages to update")
            time.sleep(60)  # wait 1 minute


def get_links_of_page(url):
    # page = requests.get(url).json()
    page = read_private_project.read_private_pages(url)
    return page["links"]


def pioneer():
    """
    Activates the pioneering mode.

    In this mode, the function identifies and extends or generates new content
    for areas in the system (e.g., Wiki pages) that are uncharted or incomplete.
    This can be particularly useful for addressing "red links" or similar gaps in the system.

    """
    print("# Pioneer mode")
    START_URL = ()
    links = []
    if args.pioneer_loop_private:
        links.extend(
            get_links_of_page(f"https://scrapbox.io/api/pages/omni-private/entry")
        )
    else:
        links.extend(
            get_links_of_page(
                f"https://scrapbox.io/api/pages/nishio/%E2%9C%8D%EF%B8%8F%F0%9F%A4%96"
            )
        )

    pages_to_update = []
    for link in links:
        link = quote_plus(link.replace(" ", "_"))

        # project may change
        project = PROJECT
        if args.pioneer_loop_private:
            project = "omni-private"
        url = f"https://scrapbox.io/api/pages/{project}/{link}"

        try:
            page = read_private_project.read_private_pages(url)
        except Exception as e:
            print("error:", e, "url:", url)
            continue

        lines = [x["text"] for x in page["lines"]]
        title = page["title"]
        if has_ai_generated_contents(lines):
            continue

        print(link)
        # show detail
        show_search_result = False
        do_direct_link = False
        if args.pioneer_loop_private:
            show_search_result = True
            do_direct_link = True

        pages_to_update.extend(
            overwrite_mode(
                title,
                lines,
                page["lines"],
                show_search_result=True,
                do_direct_link=True,
            )
        )

        # backup result
        json.dump(
            pages_to_update,
            open("pages_to_update.json", "w"),
            indent=2,
            ensure_ascii=False,
        )
    return pages_to_update


def parse_args():
    global args
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
        "--pioneer-loop-private",
        action="store_true",
        help="do pioneer-loop in private project",
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


def main():
    "Entrypoint from Github Actions"
    parse_args()
    if args.pioneer_loop or args.pioneer_loop_private:
        pioneer_loop()
        return []

    if args.pioneer:
        return pioneer()

    if args.url:
        # URL-specific overwrite, usually triggered by human
        urls = []
        if args.url == "input":
            url = input("url> ")
            print("url:", unquote(url))
            urls.append(url)
        elif args.url == "multi":
            while True:
                url = input("url> ")
                if url == "":
                    break
                print("url:", unquote(url))
                urls.append(url)
        else:
            urls.append(args.url)

        result = []
        for url in urls:
            prev_title, prev_lines = read_note_from_scrapbox(url)
            result.extend(overwrite_mode(prev_title, prev_lines))
        return result

    pages_to_update = []

    # disabled multiheads (2023-11-02)
    if 0:
        if not args.url:
            # when url is specified, user want to update the page only
            pages_to_update.extend(multiheads())

    return pages_to_update


if __name__ == "__main__":
    # main() is entrypoint from Github Actions
    pages = main()
    scrapbox_io.write_pages(pages)
    print("write ok")

    # parse_args()
    # print(fill_with_related_fragments(1000, "test", N=10, ng_list=[]))
