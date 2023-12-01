from dotenv import load_dotenv

load_dotenv()
import openai
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
import json
import os

pre_prompt = """"""

# TODO: consider splitting it into specs
# TODO: think about the experiment setup with and without the additional data
# TODO: when to show it variations and when not
prompt_template = """
I am developing a deep-learning model for article classification, which needs to categorize articles into 70 distinct classes. However, my training dataset suffers from class imbalance, with some categories having abundant samples while others lack sufficient representation. To enhance the model's ability to differentiate between classes, I aim to address this issue by generating synthetic articles that incorporate new domain-specific vocabulary and phrasing relevant to the respective categories.
Below you will find the article that you are meant to rewrite and its assigned class. Please rewrite it in a way that incorporates fresh class-specific knowledge, vocabulary and phrasing. It is essential that you maintain the integrity of the original category. Do you understand?

Category: {category}
Title: {title}
Article:
{article}
"""

# Set up your OpenAI API credentials
AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_SERVICE") or ""
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY") or ""
AZURE_OPENAI_MODEL_NAME = os.environ.get("AZURE_OPENAI_MODEL_NAME") or "gpt-35-turbo"
AZURE_OPENAI_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME") or "chat"

OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION") or ""

openai.api_type = "azure"
openai.api_base = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com/"
openai.api_version = OPENAI_API_VERSION
openai.api_key = AZURE_OPENAI_API_KEY


def load_articles(filename):
    with open(filename) as file:
        json_list = list(file)
        for json_str in json_list:
            article = json.loads(json_str)
            yield article["id"], article['title'], article['text'], article['category']


def generate_response(prompt):
    messages = langchain_chatmsgs_to_openaimsgs(prompt)
    completion = openai.ChatCompletion.create(engine="chat", messages=messages, temperature=0.7, stop=None)
    return completion


def langchain_chatmsgs_to_openaimsgs(messages):
    openai_msgs = []
    for msg in messages:
        new_msg = {}
        if msg.type == "system":
            new_msg["role"] = "system"
        elif msg.type == "human":
            new_msg["role"] = "user"
        elif msg.type == "ai":
            new_msg["role"] = "assistant"
        new_msg["content"] = msg.content
        openai_msgs.append(new_msg)
    return openai_msgs


def construct_input(article, category, title):
    messages = [SystemMessagePromptTemplate.from_template(pre_prompt), HumanMessagePromptTemplate.from_template(prompt_template)]
    chat_prompt = ChatPromptTemplate.from_messages(messages)
    chat_prompt = chat_prompt.format_prompt(category=category, title=title, article=article)
    return chat_prompt.to_messages()


def main():
    filename = '../data/raw_data.jsonl'
    articles = load_articles(filename)

    for _ in range(5):  # Repeat the process 5 times
        for id, title, text, category in articles:
            category = 'Drones'  # Replace with the relevant category for the article
            messages = construct_input(text, category, title)
            response = generate_response(messages)
            print(response.choices[0].message.content)


if __name__ == '__main__':
    main()
