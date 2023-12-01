import requests
from newspaper import Article
from dotenv import dotenv_values

from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

config = dotenv_values(".env")
OPEN_AI_API = config["OPEN_AI_API"]
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPEN_AI_API)

article_url = "https://www.howzat.com/football/football-rules-and-regulations.html"
session = requests.Session()

article_title = ""
article_text = ""
messages = [SystemMessage(content="Hello, you are a summarizer assistant. I can summarize online articles.")]



try:
    response = session.get(article_url)

    if response.status_code == 200:
        article = Article(article_url)
        article.download()
        article.parse()

        article_title = article.title
        article_text = article.text

        # template = """You are a very good assistant that summarizes online articles.

        # Here's the article you want to summarize.

        # ==================
        # Title: {article_title}

        # {article_text}
        # ==================

        # Write a summary of the previous article.
        # """

        messages.append(HumanMessage(content="I want to summarize the below article."))
        messages.append(HumanMessage(content=f"Title: {article_title}"))
        messages.append(HumanMessage(content=article_text))

        
        # prompt = template.format(article_title=article_title, article_text=article_text)
        # human_message = [HumanMessage(content=prompt)]
        # messages.append(human_message)

        response = chat(messages)
        print(response)
        


    else:
        print(f"Error: {response.status_code}")

except Exception as e:
    print(f"Error: {e}")

