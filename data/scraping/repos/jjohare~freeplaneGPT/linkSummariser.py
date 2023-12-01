import openai
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from typing import List
import time
import tqdm

# Retrieve OpenAI API key from file
with open("openai.key", "r") as key_file:
    openai.api_key = key_file.read().strip()

def retrieve_page_text(url: str) -> str:
    driver = get_driver()
    try:
        driver.get(url)
    except Exception as e:
        print(f"Error visiting URL {url}: {e}")
        return ""

    time.sleep(2)  # Wait for the page to load

    page_text = ""
    try:
        page_text += driver.find_element(By.TAG_NAME, 'body').text
        iframes = driver.find_elements(By.TAG_NAME, 'iframe')
        for iframe in iframes:
            driver.switch_to.frame(iframe)
            page_text += " " + driver.find_element(By.TAG_NAME, 'body').text
            driver.switch_to.default_content()
    except Exception as e:
        print(f"Error retrieving text: {e}")

    driver.quit() # Make sure to quit driver after retrieving page text

    return page_text


def read_links_from_file(file_path: str) -> List[str]:
    with open(file_path, "r") as file:
         links = [line.strip() for line in file.readlines()]
    return links

def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    # Additional options to try to avoid the "DevToolsActivePort file doesn't exist" error
    chrome_options.add_argument("--remote-debugging-port=9222")  # This line may be helpful
    chrome_options.add_argument("--single-process")
    chrome_options.add_argument("--disable-software-rasterizer")

    webdriver_service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)
    return driver

def process_link(link: str):
    try:
        page_text = retrieve_page_text(link)
        summary = summarize(page_text)
        return f"<node TEXT=\"{summary}\" LINK=\"{link}\"/>\n\n"
    except Exception as e:
        print(f"An error occurred while processing link {link}: {e}")
        return None

def summarize(text: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please summarize the following text in less than 500 words: {text}"}
        ]
    )
    return response['choices'][0]['message']['content']

def write_summary_to_html(summaries: List[str], output_file: str):
    """
    Writes the summaries to an HTML file in the specified format.
    """
    with open(output_file, "w") as file:
        for summary in summaries:
            file.write(summary)

if __name__ == "__main__":
    links_file = "links.txt"
    output_file = "output.md"
    links = read_links_from_file(links_file)

    # We use a simple for loop to process links sequentially
    summaries = []
    for link in tqdm.tqdm(links):
        summaries.append(process_link(link))
        time.sleep(1)  # Pause for 1 second

    summaries = list(filter(None, summaries))  # Remove None values from the list

    write_summary_to_html(summaries, output_file)
