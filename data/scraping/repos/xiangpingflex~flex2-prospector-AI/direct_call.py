import openai
from dotenv import load_dotenv

from assistant.model.knowledge_base import KnowledgeBase
from assistant.model.prompts.reply_prompt import generate_reply_prompt
import os

load_dotenv()
openai.api_key = os.environ.get("OPEN-API-KEY")
vec_db = KnowledgeBase().create_knowledge_base()


def retrieve_info(vec_db, query):
    similar_response = vec_db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array


message = "How much is the Flex membership fee?"
page_contents_array = retrieve_info(vec_db, message)
print(page_contents_array)


def call_gpt():
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=generate_reply_prompt(message, page_contents_array),
        # max_tokens=max_tokens,
        temperature=0.9,
        # n=num_completions,
    )
    return response["choices"][0]["message"]["content"]


result = call_gpt()
print(result)
