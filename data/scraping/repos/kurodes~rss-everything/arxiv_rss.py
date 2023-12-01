import feedparser
import feedgenerator
from datetime import datetime
import openai
import re
from time import sleep
from typing import List

OPENAI_API_KEY = open("/home/scripts/openai_api_key").read().strip()
openai.api_key = OPENAI_API_KEY
# openai.api_key = "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"

rss_file = '/home/files/arxiv.rss'
# rss_file = "/Users/kuro/Desktop/coding/rss-everything/n8n_docker/files/arxiv.rss"

arxiv_urls = [
    "http://export.arxiv.org/rss/cs.DB",
    "http://export.arxiv.org/rss/cs.DC",
    "http://export.arxiv.org/rss/cs.NI",
    "http://export.arxiv.org/rss/cs.OS"
]

def getCompletion(messages):
    try:
       completion = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=messages
        )
       response = completion.choices[0].message.content
    except Exception as e:
        response = str(e)
    return response


def main(urls: List[str]):
    input_feeds = []
    for url in urls:
        input_feeds.append(feedparser.parse(url))

    output_feed = feedgenerator.Rss201rev2Feed(
        title="CS updates on arXiv.org",
        link="'http://arxiv.org/'",
        description="'Computer Science updates on the arXiv.org e-print archive'",
        # lastBuildDate will be auto generated, equal to the latest item's pubdate
        lastBuildDate=None,
        image="https://arxiv.org/icons/sfx.gif"
    )

    # deduplicate inside each excution
    unique_ids = set()

    for feed in input_feeds:
        for item in feed.entries:
            if item.link not in unique_ids:

                # filter
                if not any(s in item.title for s in ["cs.DB", "cs.DC", "cs.NI", "cs.OS"]):
                    continue

                # chatgpt
                messages = [{"role": "system", "content" : "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-03-02"}]
                messages.append({"role": "user", "content": f"Translate this title of a computer science paper into Chinese: {item.title}"})
                title_translation = getCompletion(messages)
                messages.append({"role": "user", "content": f"As a new PhD candidate, I'm having difficulty understanding a computer science paper with the abstract provided below. Please help me by:\n1. Translating the abstract into Chinese.\n2. For each technical term used in the abstract, provide a detailed explanation in English, followed by its translation into Chinese. Organize your response in a list format, with each item containing the technical term, its explanation, and its Chinese translation.\n\n\"\"\"\n{item.summary}\n\"\"\""})
                abs_explanation = re.sub(r'\n', '<br>', getCompletion(messages))
                formated_description = f"<p>{title_translation}</p><p>{item.author}</p><p>{item.summary}</p><p>{abs_explanation}</p>"
                # formated_description = item.summary

                # add to feed
                output_feed.add_item(
                    title=item.title,
                    link=item.link,
                    description=formated_description,
                    author_name=item.author,
                    # A string that uniquely identifies the item.
                    unique_id=item.link,
                )
                unique_ids.add(item.link)

    # xml_string = output_feed.writeString("utf-8")
    # print(xml_string)

    with open(rss_file, 'w') as fp:
        output_feed.write(fp, 'utf-8')
        fp.close()

if __name__ == "__main__":
    main(arxiv_urls)
