import os
import dotenv

# load env variables
dotenv.load_dotenv()

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)
from openai import OpenAI
import chromadb
import requests
from selenium import webdriver
from bs4 import BeautifulSoup
import time
from sentence_transformers import SentenceTransformer

client = OpenAI()


def test_read_html(url):
    with open("index.html", "r") as f:
        return f.read()


if __name__ == "__main__":
    url = "https://www.nike.com/"

    # set up vector database
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="storage")

    # Create a WebDriver instance (for Chrome)
    # driver = webdriver.Chrome()

    # get html text
    # driver.get(url)
    # time.sleep(3)

    # html_text = driver.page_source

    html_text = test_read_html(url)
    soup = BeautifulSoup(html_text, "html.parser")

    # Approach: find all interactive elements
    # buttons, links, input, anything with text as children

    # 1. find all buttons
    button_elements = soup.find_all(["button", "input"], type=["button", "submit"])

    # 2. find all inputs
    input_elements = soup.find_all("input")

    # 3. find all links
    link_elements = soup.find_all("a")

    # 4. find all text data
    text_elements = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])

    elements = [*button_elements, *input_elements, *link_elements, *text_elements]
    elements_html = [element.prettify() for element in elements]

    # for element in elements_html:
    #     print(element)

    # create embeddings and store in vector database
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    embeddings = model.encode(elements_html)

    # collection.add(embeddings=embeddings, documents=elements_html)

    # with open("index.html", "w") as f:
    #     f.write(soup.prettify())

    # split html
    # html_splitter = RecursiveCharacterTextSplitter.from_language(
    #     language=Language.HTML, chunk_size=2048, chunk_overlap=0
    # )
    # html_docs = html_splitter.create_documents([html_text])

    # print(html_docs)


def run_conversation():
    # Step 1: Find the best css element to execute the task
    messages = [
        {
            "role": "user",
            "content": """You are given a list of html elements from a web page. Return which html element I should use on to achieve my goal. Format response should only be the entire HTML element. 
            GOAL: I want to search for "shoes
            HTML: <button aria-label="Reset Search" class="pre-clear-search ncss-btn pr0-sm z2 d-sm-h" data-var="vsClearSearch" type="click_searchClear">,<input aria-controls="VisualSearchSuggestionsList" aria-expanded="false" aria-label="Search Products" aria-owns="VisualSearchSuggestionsList" autocomplete="off" class="pre-search-input headline-5" data-var="vsInput" id="VisualSearchInput" name="search" placeholder="Search" role="combobox" tabindex="0" type="text"/>
            """,
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "click",
                "description": "Click on an html element",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "html": {
                            "type": "string",
                            "description": "html code",
                        },
                    },
                    "required": ["html"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    print(response)

    return response


print(run_conversation())
