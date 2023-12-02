import re
import sqlite3
from collections import deque
from typing import Dict, List

import requests
from langchain.chat_models import ChatOpenAI

from src import langchain_summarize
from src.common import city_table_connection
from src.model import CoordinatesQueryResponse, WikiCategoryResponse, WikiPageResponse

_WIKIVOYAGE_URL = "https://en.wikivoyage.org/w/api.php"
_WIKIPEDIA_URL = "https://en.wikipedia.org/w/api.php"


def _category_query_params(category: str) -> Dict:
    return {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": category,
        "cmlimit": 500,
    }


def _page_query_params(page_title: str) -> Dict:
    return {
        "action": "query",
        "format": "json",
        "titles": page_title,
        "prop": "extracts",
        "explaintext": True,
        "inprop": "url",
    }


def _coordinate_query_params(city: str) -> Dict:
    return {
        "action": "query",
        "format": "json",
        "titles": city,
        "prop": "coordinates",
    }


def _create_url_from_page_id(page_id: int) -> str:
    return f"https://en.wikivoyage.org/?curid={page_id}"


def parse_category_page(category: str) -> List[str]:
    """
    Create a queue that goes down all subcategories of the Germany category
    Process the queue to get all pages within all subcategories
    """
    categories: deque = deque()
    categories.append(f"Category:{category}")

    pages = []

    category_counter: int = 0
    while categories:
        category = categories.popleft()

        response = requests.get(_WIKIVOYAGE_URL, params=_category_query_params(category))  # type: ignore
        response_data = WikiCategoryResponse.parse_obj(response.json())
        for member in response_data.query.categorymembers:
            if member.ns == 14:
                categories.append(member.title)
            if member.ns == 0:
                pages.append(member.title)

        category_counter += 1
        print(f"Processed {category_counter} categories")

    return pages


def _insert_city_description_in_table(
    llm: ChatOpenAI, cursor: sqlite3.Cursor, city: str, table_name: str
) -> None:
    content_response = requests.get(_WIKIVOYAGE_URL, params=_page_query_params(city))
    page_content = WikiPageResponse.parse_obj(content_response.json())

    # Extract the page content
    for _, page_info in page_content.query.pages.items():
        city = page_info.title
        page_extract = page_info.extract

        is_city = not re.search("== Regions ==", page_extract)
        if is_city:
            cursor.execute(f"SELECT city FROM {table_name} WHERE city='{city}'")
            is_city_not_present = cursor.fetchone() is None

            if is_city_not_present:
                print(f"Getting city summary for {city}.")
                city_description = langchain_summarize.gpt_summary(
                    llm, page_extract, city
                )

                print(f"Writing info for {city} city.")
                cursor.execute(
                    f"INSERT INTO {table_name} (city, description, url) VALUES (?, ?, ?)",
                    (
                        city,
                        city_description,
                        _create_url_from_page_id(page_info.pageid),
                    ),
                )


def cities_lat_lon(
    conn: sqlite3.Connection, input_table: str, output_table: str
) -> None:
    """
    Scrape Wikipedia for the same cities to get co-ordinates
    """
    conn.execute(
        f"""
        CREATE TABLE {output_table}(
            city TEXT,
            lat REAL,
            lon REAL
        )
    """
    )

    with conn:
        cursor = conn.cursor()

        cursor.execute(f"SELECT city FROM {input_table}")
        cities = cursor.fetchall()

        for city in cities:
            print(f"Querying co-ordinates for city: {city[0]}")
            content_response = requests.get(
                _WIKIPEDIA_URL, params=_coordinate_query_params(city)
            )
            city_coords_resp = CoordinatesQueryResponse.parse_obj(
                content_response.json()
            )

            for _, page in city_coords_resp.query.pages.items():
                if page.coordinates:
                    # If we don't find co-ordinates don't put in table
                    cursor.execute(
                        f"INSERT INTO {output_table} (city, lat, lon) VALUES (?, ?, ?)",
                        (city[0], page.coordinates[0].lat, page.coordinates[0].lon),
                    )


def cities_table(
    llm: ChatOpenAI,
    page_titles: List[str],
    conn: sqlite3.Connection,
    table_name: str,
) -> None:
    """
    We want city description. To start with we were doing
    crude regex which was not very good. Then we tried using
    Pythia but summaries were pretty bad. So finally, we moved
    over to ChatGPT
    """

    with conn:
        cursor = conn.cursor()

        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name}(
                city TEXT,
                description TEXT,
                url TEXT
            )
        """
        )

        for city in page_titles:
            _insert_city_description_in_table(llm, cursor, city, table_name)


if __name__ == "__main__":
    """
    This script generates city description in the cities table
    and lat lon in the cities_lat_lon table.

    They need to be run only once.
    """
    # Get all pages under the category Germany
    pages = parse_category_page(category="Germany")

    # Add city descriptions using ChatGPT
    llm = langchain_summarize.get_llm()
    conn = city_table_connection()
    cities_table(llm, pages, conn, table_name="cities")

    cities_lat_lon(conn, input_table="cities", output_table="cities_lat_lon")
    conn.close()
