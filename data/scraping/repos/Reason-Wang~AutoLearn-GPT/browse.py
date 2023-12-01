# Adopted from Significant-Gravitas.

import logging
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, urljoin
from openai import InvalidRequestError


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


# Function to sanitize the URL
def sanitize_url(url):
    return urljoin(url, urlparse(url).path)


def split_text(text, max_length=8192):
    """Split text into chunks of a maximum length"""
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


def get_response(url, timeout=10):
    try:
        # Most basic check if the URL is valid:
        if not url.startswith('http://') and not url.startswith('https://'):
            raise ValueError('Invalid URL format')

        sanitized_url = sanitize_url(url)

        response = requests.get(sanitized_url, timeout=timeout)

        # Check if the response contains an HTTP error
        if response.status_code >= 400:
            return None, "Error: HTTP " + str(response.status_code) + " error"

        return response, None
    except ValueError as ve:
        # Handle invalid URL format
        return None, "Error: " + str(ve)

    except requests.exceptions.RequestException as re:
        # Handle exceptions related to the HTTP request (e.g., connection errors, timeouts, etc.)
        return None, "Error: " + str(re)


def scrape_text(url):
    """Scrape text from a webpage"""
    response, error_message = get_response(url)
    if error_message:
        return error_message

    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


def get_text_summary(url, query, summary_model):
    """Return the results of a google search"""
    text = scrape_text(url)
    """Summarize text using the LLM model"""

    text_length = len(text)
    print(f"Text length: {text_length} characters")

    summaries = []
    chunks = list(split_text(text))

    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1} / {len(chunks)}")
        prompt = f"The following is scraped text from \"{url}\". Extract the information related to \"{query}\"\n\n{chunk}"
        try:
            summary = summary_model.generate(prompt)
        except InvalidRequestError as e:
            logging.error(f"{e}, skip this chunk.")
            continue
        except Exception as e:
            raise e

        summaries.append(summary)

    print(f"Summarized {len(chunks)} chunks.")

    combined_summary = "\n".join(summaries)
    get_final_extraction_prompt = f"The following texts are summaries of \"{url}\". Extract the information related to \"{query}\" and ignore other unrelated information.\n\n{summaries}"
    final_extraction = summary_model.generate(get_final_extraction_prompt)

    return final_extraction


def browse_website(url, query, summary_model):
    """Browse a website and return the summary and links"""
    summary = get_text_summary(url, query, summary_model)

    return summary