import os
from pprint import pprint

import openai  # for generating embeddings
import pandas as pd  # for DataFrames to store article sections and embeddings
import pinecone
import tiktoken  # for counting tokens
from bs4 import BeautifulSoup
from tqdm.auto import tqdm  # this is our progress bar

# Import the Zenpy Class
from zenpy import Zenpy

# GLOBAL VARIABLES
MAX_INPUT_TOKENS = 8191
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request
ZENDESK_API_KEY = os.environ["ZENDESK_API_KEY"]


def fetch_zendesk_sections(zenpy_client: Zenpy):
    sections = []
    for section in zenpy_client.help_center.sections():
        if section.name == "IT Queries":
            section.name = "IT"
        else:
            section.name = "HR"
        sections.append(section)
        pass
    print(f"Sections: {sections}")
    return sections


def fetch_all_zendesk_articles(zenpy_client: Zenpy):
    articles = zenpy_client.help_center.articles()
    for article in articles:
        pprint(article)
        pass
    return articles


def fetch_zendesk_articles_by_section(zenpy_client: Zenpy, sections):
    my_articles = []
    for _section in sections:
        category = "IT" if _section.name == "IT" else "HR"
        print(f"Searching for articles in section {_section.name}")
        articles = zenpy_client.help_center.sections.articles(section=_section)
        print(f"Found {len(articles)} articles in section {_section}")
        for article in articles:
            # pprint("--------------------------------------------------------------------------------------------------")
            my_articles.append((article.title, article.body, category))
            pass
    return my_articles


def create_txt_knowledge_base(articles, path: str):
    if not os.path.exists(path):
        os.mkdir(path)

    with open(f"{path}/base.txt", "w") as file:
        for article in articles:
            file.write(article[0] + "\n" + article[1] + "\n" + article[2] + "\n\n")
            pass
        pass


def num_tokens_from_text(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def clean_up_text(articles):
    cleaned_articles = []
    for title, body, category in articles:
        cleaned_body = BeautifulSoup(body, "html.parser").get_text()
        if (
            num_tokens_from_text(title.strip() + cleaned_body.strip())
            > MAX_INPUT_TOKENS
        ):
            left = body[:MAX_INPUT_TOKENS]
            right = body[MAX_INPUT_TOKENS:]
            cleaned_articles.append((title, left, category))
            cleaned_articles.append((title, right, category))
        else:
            cleaned_articles.append((title, cleaned_body, category))
    pass
    return cleaned_articles


def print_example_data(articles):
    for article in articles[:3]:
        print(article[0])
        print(article[1][:77])
        print(article[2])
        print("-" * 50)

    for article in reversed(articles[-3:]):
        print(article[0])
        print(article[1][:77])
        print(article[2])
        print("-" * 50)


def calculate_embeddings(articles):
    titles = []
    content = []
    categories = []
    embeddings = []
    for batch_start in range(0, len(articles), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = articles[batch_start:batch_end]
        titles.extend([article[0] for article in batch])
        content.extend([article[1] for article in batch])
        categories.extend([article[2] for article in batch])
        batch_text = [title + " " + body for title, body, category in batch]
        print(f"Batch {batch_start} to {batch_end - 1}")
        response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch_text)
        for i, be in enumerate(response["data"]):
            assert (
                i == be["index"]
            )  # double check embeddings are in same order as input
        batch_embeddings = [e["embedding"] for e in response["data"]]
        embeddings.extend(batch_embeddings)

    return (
        pd.DataFrame(
            {
                "titles": titles,
                "content": content,
                "categories": categories,
                "embedding": embeddings,
            }
        ),
        embeddings,
    )


def save_dataframe_to_csv(df: pd.DataFrame, path: str, filename: str):
    if not os.path.exists(path):
        os.mkdir(path)
        print(f"Created {path}")
    df.to_csv(f"{path}/{filename}", index=False)


def store_embeddings_into_pinecone(
    df: pd.DataFrame, index: pinecone.Index, email: str = "chipzstar.dev@googlemail.com"
):
    batch_size = 32  # process everything in batches of 32
    for i in tqdm(range(0, len(df), batch_size)):
        i_end = min(i + batch_size, len(df))
        batch = df[i : i + batch_size]
        embeddings_batch = batch["embedding"]
        ids_batch = [str(n) for n in range(i, i_end)]
        # prep metadata and upsert batch
        meta = [
            {"title": titles, "content": content, "category": categories}
            for titles, content, categories, embeddings in batch.to_numpy()
        ]
        to_upsert = zip(ids_batch, embeddings_batch, meta)
        # upsert to Pinecone
        index.upsert(vectors=list(to_upsert), namespace=email)
