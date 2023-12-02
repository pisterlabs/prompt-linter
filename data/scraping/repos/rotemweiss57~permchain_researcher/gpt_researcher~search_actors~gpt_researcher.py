from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from typing import Generator

from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnableLambda
import json
from langchain.schema.messages import SystemMessage

import logging
from sys import platform

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.safari.options import Options as SafariOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from duckduckgo_search import DDGS

summary_message = (
    '"""{chunk}""" Using the above text, answer the following'
        ' question: "{question}" -- if the question cannot be answered using the text,'
        " simply summarize the text in depth. "
        "Include all factual information, numbers, stats etc if available."
)
SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("user", summary_message)
])

search_message = (
'Write 4 google search queries to search online that form an objective opinion from the following: "{question}"'\
           f'You must respond with a list of strings in the following format: ["query 1", "query 2", "query 3", "query 4"]'

)

SEARCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{agent_prompt}"),
    ("user", search_message)
])

AUTO_AGENT_INSTRUCTIONS = """
        This task involves researching a given topic, regardless of its complexity or the availability of a definitive answer. The research is conducted by a specific agent, defined by its type and role, with each agent requiring distinct instructions.
        Agent
        The agent is determined by the field of the topic and the specific name of the agent that could be utilized to research the topic provided. Agents are categorized by their area of expertise, and each agent type is associated with a corresponding emoji.

        examples:
        task: "should I invest in apple stocks?"
        response: 
        {
            "agent": "💰 Finance Agent",
            "agent_role_prompt: "You are a seasoned finance analyst AI assistant. Your primary goal is to compose comprehensive, astute, impartial, and methodically arranged financial reports based on provided data and trends."
        }
        task: "could reselling sneakers become profitable?"
        response: 
        { 
            "agent":  "📈 Business Analyst Agent",
            "agent_role_prompt": "You are an experienced AI business analyst assistant. Your main objective is to produce comprehensive, insightful, impartial, and systematically structured business reports based on provided business data, market trends, and strategic analysis."
        }
        task: "what are the most interesting sites in Tel Aviv?"
        response:
        {
            "agent:  "🌍 Travel Agent",
            "agent_role_prompt": "You are a world-travelled AI tour guide assistant. Your main purpose is to draft engaging, insightful, unbiased, and well-structured travel reports on given locations, including history, attractions, and cultural insights."
        }
    """
CHOOSE_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=AUTO_AGENT_INSTRUCTIONS),
("user", "task: {task}")
])




def get_text(soup):
    """Get the text from the soup

    Args:
        soup (BeautifulSoup): The soup to get the text from

    Returns:
        str: The text from the soup
    """
    text = ""
    tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'p']
    for element in soup.find_all(tags):  # Find all the <p> elements
        text += element.text + "\n\n"
    return text
def scrape_text_with_selenium(url: str) -> tuple[WebDriver, str]:
    """Scrape text from a website using selenium

    Args:
        url (str): The url of the website to scrape

    Returns:
        Tuple[WebDriver, str]: The webdriver and the text scraped from the website
    """
    logging.getLogger("selenium").setLevel(logging.CRITICAL)

    options_available = {
        "chrome": ChromeOptions,
        "safari": SafariOptions,
        "firefox": FirefoxOptions,
    }

    options = options_available["chrome"]()
    options.add_argument("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",)
    options.add_argument('--headless')
    options.add_argument("--enable-javascript")

    if platform == "linux" or platform == "linux2":
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--remote-debugging-port=9222")
    options.add_argument("--no-sandbox")
    options.add_experimental_option(
        "prefs", {"download_restrictions": 3}
    )
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    # Get the HTML content directly from the browser's DOM
    page_source = driver.execute_script("return document.body.outerHTML;")
    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    # text = soup.get_text()
    text = get_text(soup)

    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    return driver, text

ddgs = DDGS()

def web_search(query: str, num_results: int = 4) -> str:
    """Useful for general internet search queries."""
    print("Searching with query {0}...".format(query))
    search_results = []
    if not query:
        return json.dumps(search_results)

    results = ddgs.text(query)
    if not results:
        return json.dumps(search_results)

    total_added = 0
    for j in results:
        search_results.append(j)
        total_added += 1
        if total_added >= num_results:
            break

    return json.dumps(search_results, ensure_ascii=False, indent=4)


def split_text(text: str, max_length: int = 8192) -> Generator[str, None, None]:
    """Split text into chunks of a maximum length

    Args:
        text (str): The text to split
        max_length (int, optional): The maximum length of each chunk. Defaults to 8192.

    Yields:
        str: The next chunk of text

    Raises:
        ValueError: If the text is longer than the maximum length
    """
    paragraphs = text.split("\n")
    current_length = 0
    current_chunk = []

    for paragraph in paragraphs:
        if current_length + len(paragraph) + 1 <= max_length:
            current_chunk.append(paragraph)
            current_length += len(paragraph) + 1
        else:
            yield "\n".join(current_chunk)
            current_chunk = [paragraph]
            current_length = len(paragraph) + 1

    if current_chunk:
        yield "\n".join(current_chunk)


summary_chain = SUMMARY_PROMPT | ChatOpenAI() | StrOutputParser()
chunk_and_combine = (lambda x: [{
    "question": x["question"],
    "chunk": chunk
} for chunk in split_text(x["text"])]) | summary_chain.map() | (lambda x: "\n".join(x))

recursive_summary_chain = {
    "question": lambda x: x["question"],
    "chunk": chunk_and_combine
} | summary_chain

scrape_and_summarize = {
    "question": lambda x: x["question"],
    "text": lambda x: scrape_text_with_selenium(x['url'])[1],
    "url": lambda x: x['url']
} | RunnableMap({
    "summary": recursive_summary_chain,
    "url": lambda x: x['url']
}) | (lambda x: f"Source Url: {x['url']}\nSummary: {x['summary']}")

multi_search = (lambda x: [
    {"url": url.get("href"), "question": x["question"]}
    for url in json.loads(web_search(x["question"]))
]) | scrape_and_summarize.map() | (lambda x: "\n".join(x))

search_query = SEARCH_PROMPT | ChatOpenAI() |  StrOutputParser() | json.loads
choose_agent = CHOOSE_AGENT_PROMPT | ChatOpenAI() | StrOutputParser() | json.loads

get_search_queries = {
    "question": lambda x: x,
    "agent_prompt": {"task": lambda x: x} | choose_agent | (lambda x: x["agent_role_prompt"])
} | search_query


class GPTResearcherActor:

    @property
    def runnable(self):
        return (
                get_search_queries
                | (lambda x: [{"question": q} for q in x])
                | multi_search.map()
                | (lambda x: "\n\n".join(x))
        )

