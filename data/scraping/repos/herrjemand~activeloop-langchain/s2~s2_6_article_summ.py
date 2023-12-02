from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

import requests
from newspaper import Article

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
}

article_url = "https://www.nytimes.com/2023/09/23/us/politics/canada-sikh-leader-killing-intelligence.html"

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
        print("Error: ", response.status_code)

except Exception as e:
    print(e)

from langchain.schema import HumanMessage


template = """You are a very good assistant that summarizes online articles.

Here's the article you want to summarize.

==================
Title: {article_title}

{article_text}
==================

Write a summary of the previous article.
"""

prompt = template.format(article_title=article_title, article_text=article_text)

messages = [HumanMessage(content=prompt)]


from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(model="gpt-4", temperature=0)
summary = chat(messages)
print(summary.content)


#Bulleted

template = """You are an advanced AI assistant that summarizes online articles into bulleted lists.

Here's the article you need to summarize.

==================
Title: {article_title}

{article_text}
==================

Now, provide a summarized version of the article in a bulleted list format.
"""

# format prompt
prompt = template.format(article_title=article.title, article_text=article.text)

# generate summary
summary = chat([HumanMessage(content=prompt)])
print(summary.content)

# In french

template = """You are an advanced AI assistant that summarizes online articles into bulleted lists in French.

Here's the article you need to summarize.

==================
Title: {article_title}

{article_text}
==================

Now, provide a summarized version of the article in a bulleted list format, in French.
"""

# format prompt
prompt = template.format(article_title=article.title, article_text=article.text)

# generate summary
summary = chat([HumanMessage(content=prompt)])
print(summary.content)
