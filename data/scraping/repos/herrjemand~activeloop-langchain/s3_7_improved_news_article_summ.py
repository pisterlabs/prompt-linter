from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

import requests
from newspaper import Article

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Cache-Control': 'no-cache',
    'Sec-Ch-Ua': '"Brave";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"macOS"',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
}

article_url = "https://meduza.io/en/feature/2023/09/29/he-wanted-to-belong-there"

session = requests.Session()

article_title = ""
article_text = ""

try:
    response = session.get(article_url, headers=headers, timeout=10)

    if response.status_code == 200:
        article = Article(article_url)
        article.download()
        article.parse()

        article_title = article.title
        article_text = article.text
    else:
        print("Error: ", response.text)
        print("Error: ", response.status_code)

except Exception as e:
    print(e)

from langchain.schema import HumanMessage

template = """
As an advanced AI, you've been tasked to summarize online articles into bulleted points. Here are a few examples of the articles you'll be summarizing:

Example 1:
Original Article: 'The Effects of Climate Change
Summary:
- Climate change is causing a rise in global temperatures.
- This leads to melting ice caps and rising sea levels.
- Resulting in more frequent and severe weather conditions.

Example 2:
Original Article: 'The Evolution of Artificial Intelligence
Summary:
- Artificial Intelligence (AI) has developed significantly over the past decade.
- AI is now used in multiple fields such as healthcare, finance, and transportation.
- The future of AI is promising but requires careful regulation.

Now, here's the article you need to summarize.

==================
Title: {article_title}

{article_text}

==================

Please provide a summarized version of the article in bulleted list format.
"""

prompt = template.format(article_title=article_title, article_text=article_text)

messages = [HumanMessage(content=prompt)]

from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(model_name="gpt-4", temperature=0)

summary = chat(messages)
print(summary)

# Output parsers

from langchain.output_parsers import PydanticOutputParser
from pydantic import validator, BaseModel, Field
from typing import List


class ArticleSummary(BaseModel):
    title: str = Field(description="Title of the article")
    summary: List[str] = Field(description="Summary of the article")

    @validator("summary", allow_reuse=True)
    def has_three_or_mode_lines(cls, v):
        assert len(v) >= 3, "Summary must have at least 3 bullet points"
        return v
    
parser = PydanticOutputParser(pydantic_object=ArticleSummary)

from langchain.prompts import PromptTemplate

template = """
You are a very good assistant that summarizes online articles.

Here's the article you want to summarize.

==================
Title: {article_title}

{article_text}
==================

{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["article_title", "article_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

formatted_prompt = prompt.format_prompt(article_title=article_title, article_text=article_text)

from langchain.llms import OpenAI

# instantiate model class
model = OpenAI(model_name="text-davinci-003", temperature=0.0)

# Use the model to generate a summary
output = model(formatted_prompt.to_string())

parsed_output = parser.parse(output)
print(parsed_output)