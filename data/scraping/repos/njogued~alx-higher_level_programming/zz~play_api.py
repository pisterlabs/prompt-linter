import openai
import os
from playwright.sync_api import sync_playwright

jd = input("Enter the link to the job: ")
with sync_playwright() as s:
    browser = s.firefox.launch()
    page = browser.new_page()
    page.goto(jd)
    summary = page.query_selector_all('p')
    print(summary)
    browser.close()
# text = input("Enter the prompt: ")
text = "In 100 words or less, explain the domestication of cats"
# length = input("Max words output: ")
length = 1000
print(os.getenv("OPENAI_API_KEY"))
openai.api_key="API-KEY"
response = openai.Completion.create(
    model="text-davinci-003",
    prompt = text,
    temperature=0.7,
    max_tokens=length
)
prompt = response.choices[0].text
print(prompt)
