import openai
import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

openai.api_key = os.environ.get('LIFE_OPS_OPENAI_KEY')

def review_request(content, life_document, model="gpt-4"):

    chat = ChatOpenAI(temperature=1.0, openai_api_key=os.environ.get('LIFE_OPS_OPENAI_KEY') )

    system_prompt = f"""
You are a life coach. Your task is to review the following life task and provide feedback on it. You can provide feedback on the task itself, or on the way it is written.
You should use the following paradigms when reviewing the task:
- SMART analysis: Is the task Specific, Measurable, Achievable, Relevant, and Time-bound?
- Is the task written in the imperative mood? (e.g. "Do X" instead of "I will do X")
- (If relevant) Is the task compatible with the user's life goals (which are given in the Life Document)?
- (If relevant) Is the task compatible with the user's current projects (which are given in the Life Document)?
- (If relevant) Is the task compatible with the user's current habits (which are given in the Life Document)?
- Is the task able to be incorporated in the Getting Things Done (GTD) system?

The current Life Document is as follows:
---
{life_document}
---

Your response should be in the following format (including the specific words in the brackets):
---
[REVIEW]: Your review here
[APPROVED]: True/False
---
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Review the following life task: {content}\n\nFeedback:")
    ]

    response = chat(messages)

    review = response.content.split("[REVIEW]: ")[1].split("[APPROVED]: ")[0]
    approved = response.content.split("[APPROVED]: ")[1] == "True" if "[APPROVED]: " in response.content else True

    review = review.replace("\n", "\n\n")

    return review, approved

def ai_merge(merge_request, current_content, model="gpt-4"):

    chat = ChatOpenAI(temperature=1.0, openai_api_key=os.environ.get('LIFE_OPS_OPENAI_KEY') )

    print(current_content)

    system_prompt = """
You are a merge manager assistant. Your task is to take a Life Document, which is a markdown document, and merge in 'commits' to it. 
These could be new project ideas, new habits that the user wants to build, new life goals. You shouldn't delete anything in the Life Document, but you could merge a new goal 
with an existing one, if that makes sense. Use your own expertise, as an expert in life coaching and Life Document management.
"""

    user_prompt = f"""
The current Life Document is as follows:
---
{current_content}
---

The new merge is as follows:
---
{merge_request}
---

Merged Life Document:
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    merged = chat(messages)

    return merged.content

def send_messages(chat_history, model="gpt-3.5-turbo"):

    return openai.ChatCompletion.create(
        model=model,
        messages=chat_history,
        stream=True,
        max_tokens=300,
    )