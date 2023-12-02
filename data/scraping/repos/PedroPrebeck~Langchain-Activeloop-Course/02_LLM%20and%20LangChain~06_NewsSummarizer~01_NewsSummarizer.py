from dotenv import load_dotenv
import re

load_dotenv()

# variable_pattern = r"\{([^}]+)\}"
# input_variables = re.findall(variable_pattern, template)

import json
import requests
from newspaper import Article

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36"
}

article_url = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"

session = requests.Session()

try:
    response = session.get(article_url, headers=headers, timeout=10)

    if response.status_code == 200:
        article = Article(article_url)
        article.download()
        article.parse()

        print("Article downloaded successfully")
        # print(f"Title: {article.title}")
        # print(f"Text: {article.text}")

    else:
        print(f"Failed to fetch article at {article_url}")

except Exception as e:
    print(f"Error occurred while fetching article at {article_url}: {e}")

from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage

article_title = article.title
article_text = article.text

system_message = """
You are a very good assistant that summarizes online articles into bulled list. You only respond with the summarized list.

The next message you receive will be from an user containing the article you want to summarize.

The article you want to summarize is:
"""

template_human = """
==================
Title: {article_title}

{article_text}
==================
"""

input_variables = re.findall(r"\{([^}]+)\}", template_human)

# prompt = PromptTemplate(template=template_human, input_variables=input_variables)
prompt = template_human.format(article_title=article.title, article_text=article.text)

messages = [SystemMessage(content=system_message), HumanMessage(content=prompt)]

from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(model_name="gpt-4", temperature=0)

summary = chat(messages)
print(summary.content)
