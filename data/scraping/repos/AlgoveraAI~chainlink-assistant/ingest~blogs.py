import re
import time
import pickle
from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain.docstore.document import Document
from concurrent.futures import ProcessPoolExecutor
from config import get_logger, DATA_DIR, MAX_THREADS
from ingest.utils import (
    remove_prefix_text,
    extract_first_n_paragraphs,
    get_description_chain,
    get_driver,
)

logger = get_logger(__name__)
driver = get_driver()


def close_popup(driver):
    try:
        close_btn = driver.find_element(By.XPATH, "/html/body/div[4]/a[2]")
        close_btn.click()
        time.sleep(2)  # give it a moment to close
    except Exception as e:
        # if we can't find the popup close button, just continue
        pass


def click_load_more_button(driver, attempts=5):
    try:
        close_popup(driver)

        wait = WebDriverWait(driver, 10)
        element = wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, "/html/body/div[1]/div/section/div/div/div[3]/div[2]/a")
            )
        )
        element.click()
        return True
    except Exception as e:
        if attempts > 0:
            time.sleep(5)
            return click_load_more_button(driver, attempts - 1)
        else:
            logger.error(f"Failed to click on 'load more'. Error: {str(e)}")
            return False


def get_blog_urls():
    urls = set()
    try:
        driver.maximize_window()
        driver.get(
            "https://blog.chain.link/?s=&categories=32&services=&tags=&sortby=newest"
        )
        time.sleep(3)
        for i in range(200):
            soup = BeautifulSoup(driver.page_source, "html.parser")
            blogs = [post.a["href"] for post in soup.findAll("div", class_="post-card")]
            urls |= set(blogs)

            if not click_load_more_button(driver):
                break

            if i % 10 == 0:
                logger.info(f"Scraped {len(urls)} blog urls")
    except Exception as e:
        logger.error(f"Error scraping blog urls: {e}")

    return urls


def to_markdown(pair):
    url, soup = pair
    output = ""
    try:
        try:
            sub_soup = soup.find("h1", id="post-title")
            heading_level = int(sub_soup.name[1:])
            output += f"{'#' * heading_level} {sub_soup.get_text()}\n\n"
        except:
            sub_soup = soup.find("h1")
            heading_level = int(sub_soup.name[1:])
            output += f"{'#' * heading_level} {sub_soup.get_text()}\n\n"

        sub_soup_2 = soup.find("div", class_="post-header")
        if not sub_soup_2:
            sub_soup_2 = soup.find("article", class_="educational-content")

        for element in sub_soup_2.find_all(
            [
                "p",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "ul",
                "li",
                "blockquote",
                "code",
                "pre",
                "em",
                "strong",
                "ol",
                "dl",
                "dt",
                "dd",
                "hr",
                "table",
                "thead",
                "tbody",
                "tr",
                "th",
                "td",
                "sup",
                "sub",
                "abbr",
            ]
        ):
            if element.name == "p":
                output += f"{element.get_text()}\n\n"
            elif element.name.startswith("h"):
                try:
                    heading_level = int(element.name[1:])
                    output += f"{'#' * heading_level} {element.get_text()}\n\n"
                except:
                    pass
            elif element.name == "ul":
                for li in element.find_all("li"):
                    output += f"- {li.get_text()}\n"
                output += "\n"
            elif element.name == "li":
                output += f"- {element.get_text()}\n"
            elif element.name == "blockquote":
                output += f"> {element.get_text()}\n\n"
            elif element.name == "code":
                output += f"`{element.get_text()}`"
            elif element.name == "pre":
                output += f"```\n{element.get_text()}\n```\n\n"
            elif element.name == "em":
                output += f"*{element.get_text()}*"
            elif element.name == "strong":
                output += f"**{element.get_text()}**"
            elif element.name == "ol":
                for li in element.find_all("li"):
                    output += f"1. {li.get_text()}\n"
                output += "\n"
            elif element.name == "dl":
                for dt, dd in zip(element.find_all("dt"), element.find_all("dd")):
                    output += f"{dt.get_text()}:\n{dd.get_text()}\n"
                output += "\n"
            elif element.name == "hr":
                output += "---\n\n"
            elif element.name == "table":
                table_text = element.get_text(separator="|", strip=True)
                output += f"{table_text}\n\n"
            elif element.name == "thead":
                output += f"{element.get_text()}\n"
            elif element.name in ["tbody", "tr", "th", "td"]:
                pass  # Ignore these elements
            elif element.name == "sup":
                output += f"<sup>{element.get_text()}</sup>"
            elif element.name == "sub":
                output += f"<sub>{element.get_text()}</sub>"
            elif element.name == "abbr":
                output += f"<abbr title='{element.get('title', '')}'>{element.get_text()}</abbr>"

        return (url, output)

    except Exception as e:
        logger.error(f"Error processing {url}: {e}")
        return (url, "")


def fetch_url_content(url):
    try:
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        return (url, soup)
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return (url, None)


title_pattern = re.compile(r"^#\s(.+)$", re.MULTILINE)
chain_description = get_description_chain()


def process_blog_entry(blog):
    try:
        url, markdown = blog
        markdown_content = remove_prefix_text(markdown)

        titles = title_pattern.findall(markdown_content)
        title = titles[0].strip() if titles else "No Title"

        para = extract_first_n_paragraphs(markdown_content, num_para=2)
        description = chain_description.predict(context=para)

        return Document(
            page_content=markdown,
            metadata={
                "source": url,
                "source_type": "blog",
                "title": title,
                "description": description,
            },
        )
    except Exception as e:
        logger.error(f"Error processing blog entry {blog[0]}: {e}")
        return None


def get_blog(soup):
    try:
        markdown = to_markdown(soup)
        doc = process_blog_entry(markdown)
        return doc
    except Exception as e:
        logger.error(f"Error processing blog entry: {e}")
        return None


# def scrap_blogs():
#     global driver
#     driver = get_driver()
#     urls = get_blog_urls()

#     logger.info(f"Total number of blog urls: {len(urls)}")

#     # Use concurrent.futures to parallelize the fetching of URLs
#     with ProcessPoolExecutor(max_workers=MAX_THREADS) as executor:
#         soups = list(tqdm(executor.map(fetch_url_content, urls), total=len(urls)))

#     unsuccessful_urls = [url for url, soup in soups if not soup]
#     successful_soups = [(url, soup) for url, soup in soups if soup]

#     # Use concurrent.futures to parallelize the markdown conversion
#     with ProcessPoolExecutor(max_workers=MAX_THREADS) as executor:
#         blogs = list(
#             tqdm(executor.map(get_blog, successful_soups), total=len(successful_soups))
#         )

#     # # with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
#     # #     blogs_documents = list(tqdm(executor.map(process_blog_entry, blogs), total=len(blogs)))

#     # blogs_documents = [process_blog_entry(blog) for blog in tqdm(blogs, desc="Processing Blog Entries")]

#     # Remove nones
#     blogs_documents = [doc for doc in blogs if doc]

#     with open(f"{DATA_DIR}/blog_documents.pkl", "wb") as f:
#         pickle.dump(blogs_documents, f)

#     logger.info(f"Scraped blog posts")

#     return blogs_documents

def scrap_blogs():
    global driver
    driver = get_driver()
    urls = get_blog_urls()

    logger.info(f"Total number of blog urls: {len(urls)}")

    soups = [fetch_url_content(url) for url in tqdm(urls)]

    unsuccessful_urls = [url for url, soup in soups if not soup]
    successful_soups = [(url, soup) for url, soup in soups if soup]

    blogs = [get_blog(soup) for soup in tqdm(successful_soups)]

    blogs_documents = [doc for doc in blogs if doc]

    with open(f"{DATA_DIR}/blog_documents.pkl", "wb") as f:
        pickle.dump(blogs_documents, f)

    logger.info(f"Scraped blog posts")

    return blogs_documents

