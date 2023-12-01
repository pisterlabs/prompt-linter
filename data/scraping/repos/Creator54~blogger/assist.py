#!/usr/bin/env python

import os
import openai
from datetime import datetime

def generate_blog_post(context):
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    messages = [{"role": "system", "content": context}]

    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
    )
    reply = chat.choices[0].message['content']

    return reply

def get_user_input():
    topic = input("Enter topic: ")
    extra_context = input("Enter blog context: ")

    return topic, extra_context

def main():
    with open("_posts/docker/2023-07-15-docker-debugging.md", "r") as f:
        context = f.read()

    topic, extra_context = get_user_input()

    context_header = f"""
Generate an unrendered Markdown blog post on the topic of {topic} {extra_context} and its core concepts, benefits, and essential commands for engineers. Create a blog post similar to the following example:

---

"""

    context_footer = '''

---

Please generate the unrendered Markdown version of the blog post based on the given context.
'''

    response_text = generate_blog_post(context_header + context + context_footer)

    current_date = datetime.today().strftime('%Y-%m-%d')
    file_name = f"{current_date}-{topic.replace(' ', '-').lower()}.md"

    print(response_text)

    topic_folder=f"_posts/{topic}"
    os.makedirs(topic_folder, exist_ok=True)

    with open(f"{topic_folder}/{file_name}", "w") as f:
        f.write(response_text)

if __name__ == "__main__":
    main()

