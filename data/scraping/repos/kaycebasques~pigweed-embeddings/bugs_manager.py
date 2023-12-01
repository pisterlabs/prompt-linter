import os
import threading

import requests
import bs4
import openai
import playwright.sync_api as playwright

import database
import utilities


# Couldn't get the scraping working. This was the usage in main.py:
# bugs = manager.add_website_source(source_id='issues.pigweed.dev')
# bug_urls = bugs_manager.get_urls()
# bugs.pages = bug_urls
# bugs.scrape_handler = bugs_manager.scrape
# bugs.preprocess_handler = bugs_manager.preprocess
# bugs.segment_handler = False
# bugs.embed_handler = False
# print(json.dumps(bugs.pages, indent=4))



def get_urls():
    url_pattern = 'https://issues.pigweed.dev/issues?q=status:open&p={}'
    urls = []
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        # TODO: The magic number 25 is based on me just checking how many
        # pages of open bugs there are. Maybe we should have an automated
        # way of detecting this...
        for page_number in range(3):
            url = url_pattern.format(page_number)
            # TODO: Switch to multi-threaded impl...
            page.goto(url, wait_until='networkidle')
            html = page.content()
            soup = bs4.BeautifulSoup(html, 'html.parser')
            for a in soup.find_all('a'):
                href = a.get('href')
                if href is None:
                    continue
                if href.startswith('issues/'):
                    urls.append(f'https://issues.pigweed.dev/{href}')
        page.close()
        browser.close()
    return urls

def scrape(url, data):
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until='load')
        html = page.content()
        data['text'] = html
        page.close()
        browser.close()

def preprocess(url, data):
    original_text = data['text']
    soup = bs4.BeautifulSoup(original_text, 'html.parser')
    body = soup.find('body')
    if body is None:
        return
    for tag_name in ['script', 'style', 'link']:
        for useless_tag in body.find_all(tag_name):
            useless_tag.decompose()
    preprocessed_text = str(body)
    data['text'] = preprocessed_text
