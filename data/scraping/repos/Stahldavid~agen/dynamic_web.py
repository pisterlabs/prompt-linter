# # !pip install playwright
# # !playwright install

# from playwright.sync_api import sync_playwright

# def dynamic_web_scraper(url: str, actions: list) -> dict:
#     """
#     Navigates and scrapes content from dynamic web pages using PlayWright.

#     Parameters:
#     - url: The initial URL to navigate to.
#     - actions: A sequence of actions to perform on the web page.

#     Returns:
#     - A dictionary containing the scraped data.
#     """
#     results = {}
#     with sync_playwright() as p:
#         browser = p.chromium.launch()
#         page = browser.new_page()
#         page.goto(url)
        
#         for action in actions:
#             action_type = action.get('action_type')
#             selector = action.get('selector')
            
#             if action_type == "navigate":
#                 page.goto(selector)  # Here, 'selector' is the URL to navigate to
#             elif action_type == "click":
#                 page.click(selector)
#             elif action_type == "extract_text":
#                 results['text'] = page.inner_text(selector)
#             elif action_type == "extract_hyperlinks":
#                 links = page.query_selector_all(selector)
#                 results['links'] = [link.get_attribute("href") for link in links]
#             elif action_type == "wait_for_element":
#                 page.wait_for_selector(selector)
                
#         browser.close()
        
#     return results



from playwright.sync_api import sync_playwright
import openai
import os
import time
import random

# Access the API key from the environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')  # Make sure to set this environment variable
openai.api_key = openai_api_key

def api_call(messages, max_response_tokens):
    for i in range(15):
        try:
            return openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k-0613",
                messages=messages,
                temperature=0.2,
                max_tokens=max_response_tokens,
            )
        except openai.error.RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
            wait_time = 2 ** i + random.random()
            print(f"Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise
    print("Maximum number of retries exceeded. Aborting...")

def summarize_text(text):
    messages = [
        {
            "role": "system",
            "content": "You are a sophisticated AI that has the ability to summarize text. Please summarize the following text."
        },
        {
            "role": "user",
            "content": text
        }
    ]
    response = api_call(messages, max_response_tokens=600)
    summary = response["choices"][0]["message"]["content"]
    return summary

def dynamic_web_scraper(url: str, actions: list) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)

        result_str = ""

        for action in actions:
            action_type = action.get('type', '')
            value = action.get('value', '')

            if action_type == 'navigate_browser':
                page.goto(value)

            elif action_type == 'previous_page':
                page.go_back()

            elif action_type == 'click_element':
                page.click(value)

            elif action_type == 'scroll_down':
                page.eval_on_selector("body", "window.scrollTo(0, document.body.scrollHeight);")

            elif action_type == 'wait_for_element':
                page.wait_for_selector(value)

            elif action_type == 'extract_text':
                elements = page.query_selector_all(value)
                text = " ".join([element.text_content() for element in elements])
                
                # Chunk and summarize if text length > 40000
                if len(text) > 40000:
                    chunks = [text[i:i + 40000] for i in range(0, len(text), 40000)]
                    summarized_chunks = [summarize_text(chunk) for chunk in chunks]
                    text = "\n".join(summarized_chunks)
                
                result_str += f"Extracted Text from {value}: {text}\n"

            elif action_type == 'extract_hyperlinks':
                elements = page.query_selector_all(value)
                result_str += f"Extracted Links from {value}: {[element.get_attribute('href') for element in elements]}\n"

            elif action_type == 'current_page':
                result_str += f"Current Page URL: {page.url}\n"

        return result_str


