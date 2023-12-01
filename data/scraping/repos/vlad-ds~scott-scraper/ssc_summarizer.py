import re
import logging
import json
import os
from time import sleep

from dotenv import load_dotenv
import openai
from ssc_scraper import link_to_id, count_tokens

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Configure logger settings
logging.basicConfig(
    filename='logs.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def chatgpt(prompt: str) -> str:
    """
    Uses the chat endpoint of the GPT-3 API to generate a response to a prompt.
    :param prompt:
    :return:
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    # log token usage and expense
    usage = response["usage"]
    prompt_tokens = usage["prompt_tokens"]
    completion_tokens = usage["completion_tokens"]
    total_tokens = usage["total_tokens"]
    expense = round(total_tokens / 1000 * 0.002, 3)
    logging.info(f"Usage: {prompt_tokens} | {completion_tokens} | {total_tokens}. Expense: {expense}")
    return response['choices'][0]['message']['content']


prompt_post_chunk = """
This is a portion of a blog post, not the beginning or end. Your response should look like this:
'''
Key ideas
[Provide a summary in bullet points of the key ideas that are proposed in the post.]
Key learnings
[Provide a summary in bullet points of the key learnings that are proposed in the post.]
Key questions
[Provide a summary in bullet points of the key questions the author (Scott) asks himself in the post.]
'''
Never step out of this structure!
Another GPT instance will use these bullet points to create a more concise summary later. 
Always refer to the author as Scott! Never refer to him as "the author"!
Remember: write only in bullet points! Don't forget! Post:
"""


def summarize_post(url):
    """
    Summarizes a single post.
    :param url:
    :return:
    """
    print(f"Summarizing post {url}")
    link_id = link_to_id(url)

    if "open-thread" in link_id:
        print("Skipping open thread.")
        return

    if "meetup" in link_id:
        print("Skipping meetup.")
        return

    if "ssc-survey" in link_id:
        print("Skipping survey.")
        return

    folder = f"data/summaries/{link_id}"
    summary_exists = os.path.exists(f"{folder}/summary0.txt")

    if summary_exists:
        print("Post already summarized.")
        return

    try:
        with open(f"data/posts/{link_id}/post.json", 'r') as f:
            post_data = json.load(f)
    except FileNotFoundError:
        print("Post not found. Skipping.")
        return

    tokens = post_data['content_tokens']
    post_text = post_data['content']

    if tokens < 3000:
        # post is short enough to summarize in one go
        prompt = prompt_post_chunk + post_text
        sleep(1)
        summary = chatgpt(prompt)
        print(summary)
        os.makedirs(folder, exist_ok=True)
        with open(f"{folder}/summary0.txt", 'w') as f:
            f.write(summary)
    else:
        print("Chunking post.")
        chunks = [post_text[i:i + 13_000] for i in range(0, len(post_text), 15_000)]
        for i, chunk in enumerate(chunks):
            chunk_tokens = count_tokens(chunk)
            print(f"Chunk {i} has {chunk_tokens} tokens.")
            if chunk_tokens > 3500:
                raise Exception("Chunk too long. Aborting.")
            prompt = prompt_post_chunk + chunk
            sleep(1)
            summary = chatgpt(prompt)
            print(summary)
            os.makedirs(folder, exist_ok=True)
            with open(f"{folder}/summary{i}.txt", 'w') as f:
                f.write(summary)


def summarize_first_pass():
    with open("data/ssc_links.json", 'r') as f:
        links = json.load(f)

    for link in links:
        summarize_post(link)


def consolidate_summary(link_id: str):
    """
    Consolidates a summary that was split into multiple files.
    :param link_id:
    :return:
    """

    if not os.path.exists(f"data/summaries/{link_id}"):
        print("Summary not found.")
        return

    summary_files = [file for file in os.listdir(f"data/summaries/{link_id}") if "summary" in file]

    if len(summary_files) <= 1:
        print("Nothing to consolidate.")
        return

    def get_number(filename: str):
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())

    summaries_with_numbers = [file for file in summary_files if get_number(file) is not None]
    summaries_with_numbers.sort(key=get_number)

    # consolidate summary files
    partial_summaries = []
    for file in summaries_with_numbers:
        with open(f"data/summaries/{link_id}/{file}", 'r') as f:
            summary = f.read()
            partial_summaries.append(summary)

    consolidated_summary = "\n\n".join(partial_summaries)

    with open(f"data/summaries/{link_id}/summary_consolidated.txt", 'w') as f:
        f.write(consolidated_summary)


def consolidate_all_summaries():
    with open("data/ssc_links.json", 'r') as f:
        links = json.load(f)

    for link in links:
        link_id = link_to_id(link)
        consolidate_summary(link_id)


def summarize_final(link_id: str):
    """
    Finalizes a summary by removing duplicates and consolidating bullet points.
    Note: I ended up not using this because the result was too compressed.
    :param link_id:
    :return:
    """

    consolidated_path = f"data/summaries/{link_id}/summary_consolidated.txt"
    final_path = f"data/summaries/{link_id}/summary_final.txt"

    if not os.path.exists(consolidated_path):
        print("Finalization not required. Skipping")
        return

    if os.path.exists(final_path):
        print("Final summary already exists. Skipping.")
        return

    with open(consolidated_path, 'r') as f:
        consolidated_summary = f.read()

    prompt = f"""You will receive several bullet points that come from GPT summarizing separate chunks of the same blog post.
Eliminate duplicates and consolidate the bullet points in a final summary.
The final output should follow this structure:
[Key ideas]
[Provide a summary in bullet points of the key ideas that are proposed in the post.]
Key learnings
[Provide a summary in bullet points of the key learnings that are proposed in the post.]
Key questions
[Provide a summary in bullet points of the key questions the author (Scott) asks himself in the post.]
Don't go out of this structure! Here are the summaries:
"""

    prompt += consolidated_summary
    sleep(1)
    final_summary = chatgpt(prompt)
    print(final_summary)
    with open(final_path, 'w') as f:
        f.write(final_summary)


def summarize_final_all():
    with open("data/ssc_links.json", 'r') as f:
        links = json.load(f)

    for link in links:
        print(link)
        link_id = link_to_id(link)
        summarize_final(link_id)
