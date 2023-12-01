import validators
from jinja2 import Template
from urllib.parse import urlparse
from get_data import MediumArticles
import re
import argparse
from dotenv import load_dotenv
import os
import validators
from excluded_urls import EXCLUDE_URLS
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# load environment variables from .env file
load_dotenv()

# Set default values
DEFAULT_USERNAME = "justdataplease"
DEFAULT_ARTICLES_LIMIT = 0
DEFAULT_ISOLATE_ARTICLES = False
USE_GPT = False
FIXED_LAST_DATE = os.environ.get('FIXED_LAST_DATE', default=None)


def trim_url(url: str) -> str:
    try:
        url = url.split("?")[0].split("#")[0]
        if url[-1] == '/':
            url = url[0:-1]
        url_last = url.split("/")[-1]
    except Exception:
        return
    return url_last


def rescale(numbers: list, scale: tuple = (30, 70)) -> dict:
    min_value = min(numbers)
    max_value = max(numbers)
    new_min, new_max = scale

    scaled_numbers = {}
    for number in numbers:
        scaled_number = ((number - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min
        scaled_numbers[number] = scaled_number

    return scaled_numbers


def get_links(user: str, isolate_articles: bool = True, articles_limit: int = 10, reset: bool = False, fixed_last_date: str = None,
              use_gpt: bool = False) -> dict:
    a = MediumArticles(username=user, articles_limit=articles_limit, reset=reset, fixed_last_date=fixed_last_date, use_gpt=use_gpt)
    articles_dict = a.get_all_articles()

    articles = articles_dict["articles"]
    user = articles_dict["user"]

    main_counter = 0

    article_index = {}
    dataset = {}

    # Find size of article shape
    voter_count_list = [s["voter_count"] for s in articles]
    voter_count_rescaled_index = rescale(voter_count_list)

    # Create nodes for articles
    for article in articles:
        main_counter += 1
        url = article['url']
        stats_dict = article['stats_dict']
        trimmed_url = trim_url(url)
        ar = {"id": main_counter, "shape": "star", "color": "#fdfd96", "label": stats_dict['h1'][0:20], "main_title": stats_dict['h1'],
              "size": voter_count_rescaled_index[article["voter_count"]], "url": url,
              "domain": url,
              "description": stats_dict['h2'], "urls": [], "main": 1, "counter": 1, "font": {"color": "#000000", "size": 20}}
        article_index[trimmed_url] = main_counter
        dataset[main_counter] = ar
        article["counter"] = main_counter

    connections = []
    # Ids of external website domain nodes will start from 100000 (articles ids start from 1)
    counter = 100000
    already_found_index = {}

    # Create nodes for external website domains and connections between them and the articles
    for article in articles:
        already_found_in_article_index = {}

        if isolate_articles:
            already_found_index = {}

        stats_text = article['stats']
        article_id = article['counter']

        dataset[article_id]["stats"] = stats_text

        for link in article['links']:
            text = link[0]
            href = link[1]

            trimmed_href = trim_url(href)
            domain = urlparse(href).netloc
            description_url = (text or "") + "|" + (href or "")

            if href:
                # Check if it is a valid url else do not bother
                if validators.url(href) and not re.search(EXCLUDE_URLS, href):

                    found_main_article = article_index.get(trimmed_href)
                    # If this is an external website domain (dot)
                    if not found_main_article:
                        if already_found_index.get(domain):
                            id = already_found_index[domain]

                            if (not already_found_in_article_index.get(domain)) or isolate_articles:
                                dataset[id]["counter"] += 1
                                if dataset[id]["size"] <= 50:
                                    dataset[id]["size"] += 2

                            dataset[id]["urls"] = list(set(dataset[id]["urls"] + [description_url]))
                            dataset[id]["label"] = dataset[id]["label"].split("|")[0] + "|" + str(dataset[id]["counter"])

                            if isolate_articles:
                                continue
                        else:
                            counter += 1
                            id = counter
                            dataset[id] = {"id": id, "shape": "dot", "url": domain, "domain": domain, "size": 10, "label": domain.replace("www.", ""),
                                           "description": text,
                                           "main": 0, "urls": [description_url], "counter": 1}

                        already_found_index[domain] = id
                        already_found_in_article_index[domain] = id
                    else:
                        # If this is a main article (star)
                        id = found_main_article

                    connections_color = '#A7C7E7' if found_main_article else '#dbd7d7'
                    highlight_color = '#3c82ca' if found_main_article else '#9a8f8f'
                    connection_edge = {"from": id, "to": article_id, "font": {"color": "#808080", "size": 10},
                                       "color": {"color": connections_color, "highlight": highlight_color}}
                    connection_edge_tuple = (id, article_id)

                    # All connections
                    connection_tuples = [(x["from"], x["to"]) for x in connections] + [(x["to"], x["from"]) for x in connections]

                    # If there is already a connection do not recreate
                    if (connection_edge_tuple not in connection_tuples) and (article_id != id):
                        connections.append(connection_edge)

    return {"nodes": list(dataset.values()), "edges": connections,
            "user_profile": user["profile"],
            "user_image": user["info"]["image_url"]}


def render_html(username=DEFAULT_USERNAME, isolate_articles=DEFAULT_ARTICLES_LIMIT, articles_limit=DEFAULT_ARTICLES_LIMIT, fixed_last_date=FIXED_LAST_DATE,
                use_gpt=USE_GPT):
    dataset = get_links(username, isolate_articles=isolate_articles, articles_limit=articles_limit, fixed_last_date=fixed_last_date, use_gpt=use_gpt)

    # Process template and generate html
    with open('templates/template.html') as file:
        template = Template(file.read())

    output_file_name = f'output/{username.replace(".", "_")}_{"i" if isolate_articles else "m"}.html'

    with open(output_file_name, 'w', encoding='utf8') as file:
        file.write(
            template.render(data=dataset, user=username,
                            user_image=dataset["user_image"],
                            user_profile=dataset["user_profile"],
                            isolate_articles=isolate_articles))


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--username", default=DEFAULT_USERNAME, help="username to retrieve links for")
    parser.add_argument("-l", "--limit", type=int, default=DEFAULT_ARTICLES_LIMIT, help="maximum number of articles to retrieve")
    parser.add_argument("-i", "--isolate", action="store_true", default=DEFAULT_ISOLATE_ARTICLES, help="whether to isolate articles")
    parser.add_argument("-fd", "--fdate", type=str, default=FIXED_LAST_DATE, help="fixed last date to calculate last seen")
    parser.add_argument("-ai", "--ai", action="store_true", default=USE_GPT, help="use chatgpt to extract keywords and summary")
    args = parser.parse_args()

    render_html(username=args.username, isolate_articles=args.isolate, articles_limit=args.limit, fixed_last_date=args.fdate, use_gpt=args.ai)
