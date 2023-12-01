from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI
from dotenv import load_dotenv

from util.text import clean_end

load_dotenv()

llm = OpenAI(temperature=0, model="text-davinci-003")


def line_topics(text):
    query = (
        "tell me the top three topics in this text. Separated the topics with newlines with no numbers, bullet points or punctuation:\n\n"
        + text
    )
    response = llm.complete(query)
    # print('query:', query)
    # print('response:', response)
    # print('response.fields', dir(response))
    text = response.text.strip().strip().strip().strip()  # all those trailing newlines
    items = text.split("\n")
    items = [item.strip() for item in items]
    items = [i for i in items if i]
    return items


def line_title(text):
    query = "tell me a single keyword that best summarizes this paragraph: " + text
    response = llm.complete(query)
    title = response.text.strip()
    title = clean_end(title)
    return title


def para_summary(text, num="twelve"):
    query = f"Write a short and concise {num} words summary of this text:\n {text}"
    response = llm.complete(query)
    summary = response.text.strip().strip()
    return summary
