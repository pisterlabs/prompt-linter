#!/usr/bin/env python3
import sys
import argparse
import json
import re

from io import BytesIO
from datetime import datetime, timedelta
from pathlib import Path

from openai import OpenAI
from bs4 import BeautifulSoup
from PIL import Image
from jinja2 import Environment, FileSystemLoader

import yaml
import requests

# Tags and attributes to search for in the HTML response to find the article content
CONTENT_QUERIES = [
    ("article", {"class": "article-content"}),
    ("article", {}),
    ("p", {"class": "story-body-text story-content"}),
    ("div", {"class": "article-body"}),
    ("div", {"class": "content"}),
    ("div", {"class": "entry"}),
    ("div", {"class": "post"}),
    ("div", {"class": "blog-post"}),
    ("div", {"class": "article-content"}),
    ("div", {"class": "article-body"}),
    ("div", {"class": "article-text"}),
    ("div", {"class": "article-wrapper"}),
    ("div", {"class": "story"}),
    ("div", {"id": "article"}),
    ("div", {"id": "content"}),
    ("div", {"id": "entry"}),
    ("div", {"id": "post"}),
    ("div", {"id": "blog-post"}),
    ("div", {"id": "article-content"}),
    ("div", {"id": "article-body"}),
    ("div", {"id": "article-text"}),
    ("div", {"id": "article-wrapper"}),
    ("section", {"class": "article-body"}),
    ("section", {"class": "article-content"}),
]

# Tags and attributes to search for in the HTML response to find the article title
TITLE_QUERIES = [
    ("title", {}),
    ("h1", {"class": "story-body__h1"}),
    ("h1", {"class": "story-body__h1"}),
    ("h1", {"class": "entry-title"}),
    ("h1", {"class": "post-title"}),
    ("h1", {"class": "blog-post-title"}),
    ("h1", {"class": "article-title"}),
    ("h1", {"class": "entry-title"}),
    ("h1", {"class": "post-title"}),
    ("h1", {"class": "blog-post-title"}),
    ("h1", {"class": "article-title"}),
    ("h1", {"class": "entry-title"}),
]


def get_news(topic: str, since_date: datetime, api_key: str):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "from": since_date.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "relevancy",
        "apiKey": api_key,
    }
    response = requests.get(url, params=params)
    return response.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai-api-key", help="OpenAI API key", required=True)
    parser.add_argument("--news-api-key", help="News API key", required=True)
    parser.add_argument("--config", help="Path to the config YAML", required=True)
    parser.add_argument(
        "--rewrite-article",
        help="Rewrite the specified article, only article name required",
        required=False,
    )
    parser.add_argument(
        "--rewrite-prompt", help="Prompt to use for rewriting", required=False
    )
    parser.add_argument(
        "--links",
        help="Write an article based on provided links (one URL on each line)",
        required=False,
    )
    args = parser.parse_args()

    # Load the config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.rewrite_article:
        return rewrite_article(
            openai_client=OpenAI(api_key=args.openai_api_key),
            openai_model=config["openai"]["model"],
            openai_temperature=config["openai"]["temperature"],
            openai_rewrite_prompt=args.rewrite_prompt,
            article_path=Path(config["blog"]["posts"]) / args.rewrite_article.strip(),
        )

    return write_article(
        openai_api_key=args.openai_api_key,
        openai_model=config["openai"]["model"],
        openai_max_tokens=config["openai"]["max_tokens"],
        openai_temperature=config["openai"]["temperature"],
        openai_article_summary_prompt=config["openai"]["article_summary_prompt"],
        openai_final_article_prompt=config["openai"]["final_article_prompt"],
        openai_final_title_prompt=config["openai"]["final_title_prompt"],
        image_generation_model=config["openai"]["dalle_model"],
        openai_prompt_for_dalle=config["openai"]["dalle_prompt"],
        news_api_key=args.news_api_key,
        topic_to_search=config["news"]["topic"],
        max_article_age=config["news"]["max_age_in_days"],
        max_articles=config["news"]["max_articles"],
        assets_dir=config["blog"]["assets"],
        posts_dir=config["blog"]["posts"],
        post_template_path=config["blog"]["post_template"],
        attribution=config["blog"].get("attribution", True),
        provided_links=args.links,
    )


def write_article(
    openai_api_key: str,
    openai_model: str,
    openai_max_tokens: int,
    openai_temperature: float,
    openai_article_summary_prompt: str,
    openai_final_article_prompt: str,
    image_generation_model: str,
    openai_final_title_prompt: str,
    openai_prompt_for_dalle: str,
    news_api_key: str,
    topic_to_search: str,
    max_article_age: float,
    max_articles: int,
    assets_dir: str,
    posts_dir: str,
    post_template_path: str,
    attribution: bool,
    provided_links: str,
):
    if not provided_links or provided_links == "":
        date_to_search_from = datetime.now() - timedelta(days=max_article_age)
        # Get news as a JSON dictionary through the News API (newsapi.org)
        print("Searching for primary sources on subject: {}".format(topic_to_search))
        news = get_news(
            topic=topic_to_search, since_date=date_to_search_from, api_key=news_api_key
        )
        if news["status"] != "ok":
            print("Error: News API returned status code: {}".format(news["status"]))
            return 1

        article_titles_and_urls = [
            (article["title"], article["url"]) for article in news["articles"]
        ]
        print("Found {} articles".format(len(article_titles_and_urls)))
    else:
        article_urls = provided_links.split("\n")
        # We need to get the titles of the articles so to form an article_titles_and_urls list
        article_titles_and_urls = []
        for article_url in article_urls:
            try:
                response = requests.get(article_url)
            except Exception as e:
                print(
                    "Exception while getting article from URL: {}".format(article_url)
                )
                return 1  # We don't want to continue if we can't get all articles
            if response.status_code != 200:
                print(
                    "Error code {} while getting article from URL: {}".format(
                        response.status_code, article_url
                    )
                )
                return 1
            # Find the article title
            soup = BeautifulSoup(response.content, "html.parser")
            for tag, attrs in TITLE_QUERIES:
                article_title = soup.find(tag, attrs)
                if article_title is not None:
                    break
            if article_title is None:
                print(
                    "Error: Could not find article title in HTML response from URL: {}".format(
                        article_url
                    )
                )
                article_title_text = article_url[:50]
                article_title_text += "..."
            else:
                article_title_text = article_title.get_text()
                # Replace any \n, \t, etc. characters in the text with spaces
                article_title_text = " ".join(article_title_text.split())
                article_title_text = article_title_text.strip()
            article_titles_and_urls.append((article_title_text, article_url))

    max_allowed_tokens = openai_max_tokens
    characters_per_token = 4  # The average number of characters per token
    max_allowed_characters = max_allowed_tokens * characters_per_token

    summarized_articles = []
    original_articles_urls = []  # Only the titles and the URLs of the articles we use

    print("Summarizing the top-{} articles...".format(max_articles))
    for article_title, article_url in article_titles_and_urls:
        try:
            response = requests.get(article_url)
        except Exception as e:
            print("Exception while getting article from URL: {}".format(article_url))
            continue
        if response.status_code != 200:
            print(
                "Error code {} while getting article from URL: {}".format(
                    response.status_code, article_url
                )
            )
            continue
        # Find the actual article content in the HTML response using the relevant SEO tags
        soup = BeautifulSoup(response.text, "html.parser")
        for tag, attrs in CONTENT_QUERIES:
            article_content = soup.find(tag, attrs)
            if article_content is not None:
                break
        if article_content is None:
            print(
                "Error: Could not find article content in HTML response from URL: {}".format(
                    article_url
                )
            )
            continue
        # Get the text from the article content
        article_text = article_content.get_text()
        # Replace any \n, \t, etc. characters in the text with spaces
        article_text = " ".join(article_text.split())

        prompt = (
            openai_article_summary_prompt
            + "\n\n```\n"
            + article_title
            + "\n\n"
            + article_text
            + "\n```"
        )
        if len(prompt) > max_allowed_characters:
            prompt = prompt[:max_allowed_characters]

        openai_client = OpenAI(api_key=openai_api_key)
        generated_summary = get_openai_response(
            prompt=prompt,
            model=openai_model,
            temperature=openai_temperature,
            openai_client=openai_client,
        )
        summarized_articles.append(generated_summary)
        original_articles_urls.append({"url": article_url, "title": article_title})

        if len(summarized_articles) >= max_articles:
            break

    if len(summarized_articles) == 0:
        print("Error: Could not summarize any articles")
        return 1

    print("Generating the final article...")
    final_article_prompt = openai_final_article_prompt + "\n" + str(summarized_articles)
    final_article = {}
    final_article["content"] = get_openai_response(
        prompt=final_article_prompt,
        model=openai_model,
        temperature=openai_temperature,
        openai_client=openai_client,
    )

    final_title_prompt = openai_final_title_prompt + "\n" + final_article["content"]
    final_article["title"] = get_openai_response(
        prompt=final_title_prompt,
        model=openai_model,
        temperature=openai_temperature,
        openai_client=openai_client,
    )
    final_article["title"] = final_article["title"].strip('"')
    # It seems that GPT models (up to GPT-4) are very biased towards generating titles
    # that are formulated as "<generic statement>: <specific statement>"
    # It's not clear how to reliably solve this with prompting, so let's keep only
    # the specific statement, i.e. the part after the colon
    if ":" in final_article["title"]:
        final_article["title"] = final_article["title"].split(":")[1].strip()
        # Capitalize the first letter of the title
        final_article["title"] = (
            final_article["title"][0].upper() + final_article["title"][1:]
        )

    print("Generating tags for the final article...")
    generated_tags_response = get_openai_response(
        prompt="Generate 3 tags as a JSON list, use one word for each tag,"
        + 'e.g. ["tag1", "tag2", "tag3"], for the following article: \n'
        + final_article["content"],
        model=openai_model,
        temperature=openai_temperature,
        openai_client=openai_client,
    )
    generated_tags = try_loads(generated_tags_response)
    if not generated_tags:
        print(
            "Error: Could not parse generated tags response: {}".format(
                generated_tags_response
            )
        )
        print("Will ignore the tags and continue")
        generated_tags = ""
    generated_tags = [tag.lower() for tag in generated_tags]
    # Split the tags into comma-separated values
    generated_tags = ", ".join(generated_tags)
    generated_tags = "[" + generated_tags + "]"

    print("Generating the prompt for the image generation...")
    prompt_gpt_to_create_dalle_prompt = openai_prompt_for_dalle + final_article["title"]
    dalle_prompt = get_openai_response(
        prompt=prompt_gpt_to_create_dalle_prompt,
        model=openai_model,
        temperature=openai_temperature,
        openai_client=openai_client,
    )

    print(f"Generating an image based on the article with {image_generation_model}...")
    dalle_response = openai_client.images.generate(
        model=image_generation_model,
        prompt=dalle_prompt,
        n=1,
        size="1024x1024",
        response_format="url",
    )
    dalle_image_url = dalle_response.data[0].url

    print("Downloading the image...")
    response = requests.get(dalle_image_url)
    if response.status_code != 200:
        print(
            "Error code {} while getting image from URL: {}".format(
                response.status_code, dalle_image_url
            )
        )
        return 1
    image = Image.open(BytesIO(response.content))
    title_normalized = re.sub(r"[^\w\s]", "", final_article["title"])
    title_normalized = title_normalized.replace(" ", "_")
    current_date = datetime.now().strftime("%Y-%m-%d")
    image_file_name = Path("{}-{}.png".format(current_date, title_normalized))
    image_path = Path(assets_dir) / image_file_name
    image.save(image_path)

    made_with_sycophant = ""
    attribution_links = ""
    if attribution:
        # Append the links to the original articles to the final article
        made_with_sycophant = (
            "\n\nThe above article was written with the help of "
            + "[sycophant](https://github.com/platisd/sycophant) "
            + "based on content from the following articles:\n"
        )
        attribution_links = "\n".join(
            [
                "- [{}]({})".format(article["title"], article["url"])
                for article in original_articles_urls
            ]
        )

    post_title = '"' + final_article["title"].replace('"', '\\"') + '"'
    post_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S %z")
    post_tags = generated_tags
    img_path = "/" + assets_dir
    post_image = image_file_name
    image_caption = "'" + dalle_prompt + "'"
    post_content = final_article["content"] + made_with_sycophant + attribution_links
    post_content += "\n"  # Finish with a newline to be nice
    post_filename = image_file_name.with_suffix(".md")

    print("Generating the final post...")
    environment = Environment(loader=FileSystemLoader(Path(post_template_path).parent))
    template = environment.get_template(Path(post_template_path).name)
    output = template.render(
        post_title=post_title,
        post_date=post_date,
        post_tags=post_tags,
        img_path=img_path,
        post_image=post_image,
        image_caption=image_caption,
        post_content=post_content,
    )

    post_path = Path(posts_dir) / post_filename
    with open(post_path, "w") as f:
        f.write(output)

    print(
        "Done! Generated files: \n"
        + " - [Post]({})\n".format(post_path)
        + " - [Image]({})\n".format(image_path)
    )

    return 0


def rewrite_article(
    openai_client: OpenAI,
    openai_model: str,
    openai_temperature: float,
    openai_rewrite_prompt: str,
    article_path: Path,
):
    if not article_path.exists():
        print("Error: Article not found: {}".format(article_path))
        return 1

    print("Reading article from file...")
    with open(article_path, "r") as f:
        article = f.read()
        font_matter = article.split("---")[1]
        article = article.split("---")[2]

    print("Rewriting article...")
    if not openai_rewrite_prompt or openai_rewrite_prompt == "":
        openai_rewrite_prompt = (
            "Rewrite the following article but keep the last "
            + "paragraph if it includes attribution to the original "
            + "articles and the links to them:\n"
        )
    else:
        openai_rewrite_prompt = (
            "Keep the last paragraph if it includes attribution "
            + "to the original articles and the links to them.\n"
            + openai_rewrite_prompt
            + "\nThe article to rewrite is:\n"
        )
    openai_rewrite_prompt += article

    openai_response = get_openai_response(
        prompt=openai_rewrite_prompt,
        model=openai_model,
        temperature=openai_temperature,
        openai_client=openai_client,
    )

    # Replace the content of the article with the rewritten one
    with open(article_path, "w") as f:
        f.write("---")
        f.write(font_matter)
        f.write("---\n")
        f.write(openai_response + "\n")

    print("Done! Rewritten article: " + str(article_path))

    return 0


def try_loads(maybe_json: str):
    try:
        return json.loads(maybe_json, strict=False)
    except Exception as e:
        print(e)
        print("Response not a valid JSON: \n" + maybe_json)
        return None


def get_openai_response(
    prompt: str, model: str, temperature: float, openai_client: OpenAI
):
    openai_response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who summarizes news articles",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )

    return openai_response.choices[0].message.content


if __name__ == "__main__":
    sys.exit(main())
