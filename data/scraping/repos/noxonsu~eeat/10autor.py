import asyncio
from playwright.async_api import async_playwright
import html2text
import openai
import os
import json
SERPAPI_KEY = os.environ.get('SERPAPI_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
INDUSTRY_KEYWORD = os.environ.get('INDUSTRY_KEYWORD')
KEYWORD_FOR_SERP = os.environ.get('KEYWORD_FOR_SERP', INDUSTRY_KEYWORD)
BASE_GPTV = os.environ.get('BASE_GPTV','gpt-3.5-turbo-1106')
SMART_GPTV = os.environ.get('SMART_GPTV','gpt-3.5-turbo-1106')
async def fetch_page_content(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until='networkidle')
        content = await page.content()
        await browser.close()
        return content

def create_about_autor(resume, industry_query):
    
    prompt = (
            "The task involves crafting an 'About the Author' segment, using the author's resume as a base, while emphasizing their particular area of expertise. This involves reviewing the resume, distilling its essence, and composing a concise 'About the Author' piece. The process begins with the resume text and a key term, culminating in a JSON formatted 'Name' and 'Expertice' section."
            f"\n\nResume: {resume}"
            f"\n\nKey term: {industry_query}"
        )
        
    response = openai.Completion.create(
            engine=BASE_GPTV,
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )

    if response.choices[0].text:
        urls = json.loads(response.choices[0].text)
    else:
        urls = "Not found"
    return urls
    
# Использование функции
async def main():
    url = "https://www.linkedin.com/in/noxonsu/?originalSubdomain=ru"
    content = await fetch_page_content(url)
    h = html2text.HTML2Text()
    h.wrap_links = True
    text = h.handle(content)
    about_autor = create_about_autor(text, INDUSTRY_KEYWORD)
    print(about_autor)

asyncio.run(main())
