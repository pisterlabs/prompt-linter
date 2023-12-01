import os
import requests
from bs4 import BeautifulSoup
import openai
import re
import datetime
import logging
from time import sleep
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO)
MODEL_NAME = "gpt-4-1106-preview"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}


def get_response_from_prompt(model_name, prompt, system_prompt):
    """Get response from prompt using OpenAI API"""
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": prompt},
    ]

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
    )

    return response["choices"][0]["message"]["content"]


def extract_arxiv_id_from_tags(dt, dd):
    """Extract arxiv id and paper content from dt and dd tags"""
    a = dt.find("a", {"title": "Abstract"})
    arxiv_id = a.text.replace("arXiv:", "").strip()
    paper = {
        "link": "https://arxiv.org" + a["href"],
        "title": dd.find("div", {"class": "list-title"})
        .text.replace("Title:", "")
        .replace("  ", " ")
        .strip(),
        "authors": [
            a.text for a in dd.find("div", {"class": "list-authors"}).find_all("a")
        ],
        "comments": dd.find("div", {"class": "list-comments"}),
        "subjects": re.findall(
            r"\(([\w.]+)\)", dd.find("div", {"class": "list-subjects"}).text
        ),
    }
    if paper["comments"]:
        paper["comments"] = (
            paper["comments"].text.strip("\n").replace("Comments:", "").strip()
        )
    return arxiv_id, paper


def scrape_arxiv_papers(target_date=None, category="cs.LG"):
    """Scrape arxiv papers from the past week"""
    url = f"https://arxiv.org/list/{category}/pastweek?skip=0&show=1000"

    response = requests.get(url, headers=HEADERS)

    soup = BeautifulSoup(response.content, "html.parser")
    dl_idx = 0
    if target_date:
        target_date = target_date.strftime("%a, %-d %b %Y")

        date_list = list(map(lambda x: x.text, soup.find_all("h3")))
        if not date_list or target_date not in date_list:
            return {}
        dl_idx = date_list.index(target_date)
    dl = soup.find_all("dl")[dl_idx]
    dt_tags = dl.find_all("dt")
    dd_tags = dl.find_all("dd")

    assert len(dt_tags) == len(dd_tags)

    return dict(map(extract_arxiv_id_from_tags, dt_tags, dd_tags))


def fetch_arxiv_abstracts(paper_ids):
    """Fetch arxiv abstracts from paper ids"""
    url = "https://arxiv.org/abs"

    paper_abstract_map = {}
    for paper_id in paper_ids:
        response = requests.get(f"{url}/{paper_id}", headers=HEADERS)
        soup = BeautifulSoup(response.content, "html.parser")
        abstract = (
            soup.find("blockquote", {"class": "abstract mathjax"})
            .text.replace("Abstract: ", "")
            .strip()
        )
        paper_abstract_map[paper_id] = abstract
    return paper_abstract_map


def generate_paper_summary(arxiv_id, content):
    """Generate paper summary from arxiv id and paper content"""
    summary = ""
    summary += f"{arxiv_id}-{content['title']}\n"
    summary += f"{', '.join(content['authors'])}\n"
    summary += (
        f"Comments: {content['comments']}\n" if content["comments"] is not None else ""
    )
    summary += f"{'; '.join(content['subjects'])}\n"
    return summary


def largest_chunk_size(total, max_size=100):
    """Get largest chunk size given total and max size"""
    base_size = (total + max_size - 1) // max_size
    return (total + base_size - 1) // base_size


def extract_paper_ids_from_generated_text(generated_text, papers_dict):
    """Extract paper ids from generated text"""
    paper_ids = {}
    for line in generated_text.split("\n"):
        if arxiv := re.findall("(\d{4}.\d{4,6})", line):
            arxiv_id = arxiv[0]
            title = "-".join(line.split("-")[1:])
            if arxiv_id not in papers_dict or title not in papers_dict[arxiv_id]:
                for k, v in papers_dict.items():
                    if title in v:
                        paper_ids[k] = title
            else:
                paper_ids[arxiv_id] = title
    return paper_ids


def get_response_retry(model_name, prompt, system_prompt, max_retries=5):
    """Get response from prompt with retries"""
    for i in range(max_retries):
        try:
            return get_response_from_prompt(model_name, prompt, system_prompt)
        except Exception as e:
            logging.exception(f"An error occurred: {e}. Retrying...")
            sleep(2**i)  # Exponential backoff
    raise Exception("Maximum retries exceeded")


def get_batch_responses_from_prompt(
    papers_dict, base_prompt, separator="\n", max_size=100
):
    """Get batch responses from prompt"""
    contents = list(sorted(papers_dict.values()))

    chunk_size = largest_chunk_size(len(papers_dict), max_size)

    system_prompt = "You are a helpful assistant to extract useful papers on the arxiv"

    generated_text = ""
    for start in range(0, len(papers_dict), chunk_size):
        prompt = base_prompt + "\n".join(contents[start : start + chunk_size])
        generated_text += (
            get_response_retry(MODEL_NAME, prompt, system_prompt) + separator
        )

    return generated_text.strip()


def write_compilations_markdown(papers_lst, target_date):
    """Write compilations markdown file"""
    title_str = target_date.strftime("%a, %-d %b %Y")
    date_str = target_date.strftime("%Y-%m-%d")
    md_content = f"""---
title: {title_str}
date: {date_str}
---
"""
    sorted_papers = sorted(
        papers_lst, key=lambda x: float(re.match(r"(\d+\.\d+)", x)[1]), reverse=True
    )

    # Print the sorted papers with ordinal numbers
    for i, paper in enumerate(sorted_papers):
        md_content += f"{i+1}. {paper}\n\n"

    with open(f"_arxiv_today/{date_str}.md", "w") as f:
        f.write(md_content)


def is_done_already(date):
    date_str = date.strftime("%Y-%m-%d")
    filepath = f"_arxiv_today/{date_str}.md"
    return os.path.isfile(filepath)


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument(
        "--target_date",
        type=str,
        required=False,
        default=None,
        help="Target date to scrape arxiv papers",
    )
    args = argparse.parse_args()
    if args.target_date is None:
        target_date = datetime.datetime.strptime(args.target_date, "%Y-%m-%d")
    else:
        target_date = datetime.datetime.today()
        
    if is_done_already(target_date):
        logging.info(f"Already done for {target_date}")
        exit()
    logging.info(f"Target date: {target_date}")
    papers = scrape_arxiv_papers(target_date)
    logging.info(f"Number of papers: {len(papers)}")
    if len(papers) == 0:
        exit()

    papers_dict = {
        arxiv_id: generate_paper_summary(arxiv_id, content)
        for arxiv_id, content in papers.items()
    }

    base_prompt = """From given list of arXiv papers on machine learning wrapped by triple quotes, 
list of at most 15 possibly influential papers especially on reinforcement learning,
using any given information, such as title, authors, comments, subjects, etc.
Give the only result as the form as following

Format:
arxiv_id-1-title-1
arxiv_id-2-title-2

Given papers:
"""

    generated_text = get_batch_responses_from_prompt(papers_dict, base_prompt)
    logging.info(f"Generated text: {generated_text}")
    paper_ids = extract_paper_ids_from_generated_text(generated_text, papers_dict)
    paper_abstract_map = fetch_arxiv_abstracts(paper_ids)

    filtered_papers_dict = {
        k: f"{v}Abstract: {paper_abstract_map[k]}"
        for k, v in papers_dict.items()
        if k in paper_abstract_map.keys()
    }
    logging.info(f"Filtered papers: {filtered_papers_dict}")
    base_prompt = """Following is the paper published in the arXiv on machine learning.
You extract the list of possible influential papers on reinforcement learning.
Using abstracts and authorities of the authors, list at most 5 papers decreasingly ordered by potential influence
You should calculate importance score between 1.0 and 9.9 inclusive, at increments of 0.1 for each paper
Separate each paper by a empty line
The output should be given as the ordered list in markdown form like an example
Only give output without any comments
Example:
9.1 [Name of Most influential paper](link)
    * Authors: List of Authors
    * Reason 
    
9.0 [Name of second most influential paper](link)
    * Authors: List of Authors
    * Reason

Given papers:
"""
    response = get_batch_responses_from_prompt(
        papers_dict=filtered_papers_dict,
        base_prompt=base_prompt,
        separator="\n\n",
        max_size=15,
    )
    logging.info(f"Generated text: {response}")
    papers_lst = [
        x.strip()
        for x in ("\n".join(y.strip() for y in response.split("\n"))).split("\n\n")
    ]
    write_compilations_markdown(papers_lst, target_date)
    logging.info("Done")
