import os
from typing import List, Dict, Any
import openai
from dotenv import load_dotenv
from ai_devs_task import Task

load_dotenv()
ai_devs_api_key: str = os.getenv("AI_DEVS_API_KEY", "")

blogger: Task = Task(ai_devs_api_key, "blogger")
token: str = blogger.auth()
task_content: Dict[str, Any] = blogger.get_content(token)

prompt: str = """
You are a pizza master that writes a blog about pizza in polish.
Write a short paragraph about the given topic.
"""

result: List[str] = []
blog_topics: List[str] = task_content["blog"]
for topic in blog_topics:
    blog_article: openai.ChatCompletion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": topic}
        ]
    )
    result.append(blog_article.choices[0].message["content"])

answer_payload: Dict[str, List[str]] = {"answer": result}
task_result: Dict[str, Any] = blogger.post_answer(token, answer_payload)
print(task_result)
