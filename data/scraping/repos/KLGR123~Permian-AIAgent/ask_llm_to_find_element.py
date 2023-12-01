import os, sys
import requests
import time
import re
import html2text

from bs4 import BeautifulSoup
from bs4.element import NavigableString
from bs4.element import Tag

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

from llama_index import Document, GPTSimpleVectorIndex, LLMPredictor
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI


WAIT_TIME = 2

user_data_dir = "/Users/liujiarun/大四/创新基金项目/加拿大合作项目/project/kapwing_agent/agents/utils/user_data"  
options = webdriver.ChromeOptions()
options.add_argument(f"user-data-dir={user_data_dir}")
driver = webdriver.Chrome(options=options)
driver.maximize_window()
actions = ActionChains(driver)

driver.get('https://www.kapwing.com/642c11d0603909036b652b97/studio/editor')
driver.implicitly_wait(15)


PROMPT_TO_FIND_ELEMENT = """Given the HTML below, write the `By` and `value` argument to the Python Selenium function `env.find_elements(by=By, value=value)` to precisely locate the element.
The argument `By` is a string that specifies the locator strategy. `By` is usually `xpath`, `class_name` or `css_selector`.
The argument `value` is a string that specifies the locator value.
Write only the *string argument for `By`* and `value`* to the function.
HTML: {cleaned_html}
OUTPUT:"""


def get_text_from_page(entire_page=False):
    """Returns the text from the page."""

    # First, we get the HTML of the page and use html2text to convert it to text.
    if entire_page:
        html = driver.page_source
        text = html2text.html2text(html)
    else:
        text = driver.find_element(by=By.TAG_NAME, value="body").text

    # Check for iframes.
    iframes = driver.find_elements(by=By.TAG_NAME, value="iframe")
    for iframe in iframes:
        driver.switch_to.frame(iframe)
        if entire_page:
            html = driver.page_source
            text = text + "\n" + html2text.html2text(html)
        else:
            visible_text = driver.find_element(
                by=By.TAG_NAME, value="body"
            ).text
            text = text + "\n" + visible_text
        driver.switch_to.default_content()

    return text


def retrieve_information(prompt, entire_page=False):
    """Retrieves information using using GPT-Index embeddings from a page."""

    text = get_text_from_page(entire_page=entire_page)
    index = GPTSimpleVectorIndex([Document(text)])

    resp = index.query(prompt, llm_predictor=LLMPredictor(llm=OpenAI(temperature=0)))
    return resp.response.strip()


def get_llm_response(prompt, temperature=0, model=None):
    """Returns the response from the LLM model."""

    openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        temperature=temperature,
        stop=["```"]
    )

    return response["choices"][0]["message"]["content"]


def get_html_elements_for_llm():
    """Returns list of BeautifulSoup elements for use in GPT Index."""

    blacklisted_elements = set(["head", "title", "meta", "script", "style", "path", "svg", "br", "::marker"])
    blacklisted_attributes = set(["style", "ping", "src", "item*", "aria*", "js*", "data-*"])

    # Get the HTML tag for the entire page, convert into BeautifulSoup.
    html = driver.find_element(By.TAG_NAME, "html")
    html_string = html.get_attribute("outerHTML")
    soup = BeautifulSoup(html_string, "lxml")

    # Remove blacklisted items and attributes in it.
    for blacklisted in blacklisted_elements:
        for tag in soup.find_all(blacklisted):
            tag.decompose()

    # Delete the blacklisted attributes from a tag, as long as the attribute name matches the regex.
    for tag in soup.find_all(True):
        for attr in tag.attrs.copy():
            for pattern in blacklisted_attributes:
                if re.match(pattern, attr):
                    del tag[attr]

    # Remove children of elements that have children.
    elements = soup.find_all()
    [ele.clear() if ele.contents else ele for ele in elements if ele.contents]

    # Then remove any elements that do not have attributes, e.g., <p></p>.
    elements = [ele for ele in elements if ele.attrs]
    return elements


def ask_llm_to_find_element(element_description):
    """Clean the HTML from driver, ask GPT-Index to find the element,
    return Selenium code to access it. Return a GPTWebElement."""

    # Set up a dict that maps an element string to its object and its source iframe. Shape looks like:
    # element_string => {"iframe": iframe, "element": element_obj}.
    elements_tagged_by_iframe = {}

    # First, get and clean elements from the main page.
    elements = get_html_elements_for_llm()
    elements_tagged_by_iframe.update({ele.prettify(): {"iframe": None, "element": ele} for ele in elements})

    print(elements)

    # Then do it for the iframes.
    iframes = driver.find_elements(by=By.TAG_NAME, value="iframe")
    for iframe in iframes:
        break
        driver.switch_to.frame(iframe)
        elements = get_html_elements_for_llm()
        elements_tagged_by_iframe.update({ele.prettify(): {"iframe": iframe, "element": ele} for ele in elements})

    # Create the docs and a dict of doc_id to element, 
    # which will help us find the element that GPT Index returns.
    
    docs = [Document(element.prettify()) for element in elements]

    doc_id_to_element = {doc.get_doc_id(): doc.get_text() for doc in docs}
    index = GPTSimpleVectorIndex(docs)

    query = f"""Find element that matches description: {element_description}. 
    If no element matches, return 'NO_RESPONSE_TOKEN'. 
    Please be as succinct as possible, with no additional commentary.
    """

    resp = index.query(query, llm_predictor=LLMPredictor(llm=OpenAI(temperature=0)))
    doc_id = resp.source_nodes[0].doc_id

    resp_text = resp.response.strip()

    if 'NO_RESPONSE_TOKEN' in resp_text:
        print("GPT-Index could not find element. Returning None.")
        return None
    else:
        print(f"Asked GPT-Index to find element. Response: {resp_text}")
        return resp_text

    # Find the iframe that the element is from.
    # found_element = doc_id_to_element[doc_id]
    # iframe_of_element = elements_tagged_by_iframe[found_element]["iframe"]

    # Get the argument to the find_element_by_xpath function.
    # print(f"Found element: {found_element}")
    # prompt = PROMPT_TO_FIND_ELEMENT.format(cleaned_html=found_element)
    # llm_output = get_llm_response(prompt=prompt, temperature=0).strip().replace('"', "")

    # return llm_output

    # Switch to the iframe that the element is in.
    # if iframe_of_element is not None:
        # driver.switch_to.frame(iframe_of_element)

    # element = driver.find_element(by="xpath", value=llm_output)

    # Switch back to default_content.
    # self.driver.switch_to.default_content()

    # return GPTWebElement(element, iframe=iframe_of_element)
    # return element, iframe_of_element


if __name__ == "__main__":

    # text = get_text_from_page(entire_page=True)
    # text = retrieve_information("Search", entire_page=False)
    # print('Retrieved Info: ', text)

    ask_llm_to_find_element("Subtitles")
    # print(text)

    print('Testing Successfully!')
    driver.quit()
