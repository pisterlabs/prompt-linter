import os
import openai
from dotenv import load_dotenv

load_dotenv()

# API Keys
openai.api_key = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = "gpt-4"


def generate_content(blog_title, blog_title_prompt):
    """
    Generates content for a blog post using OpenAI API.

    :param blog_title: str, the title of the blog post.
    :return: str, generated content for the blog post.
    """
    with open("src/prompts/body.txt", "r", encoding="UTF-8") as body_prompt_file:
        prompt = body_prompt_file.read()

    prompt = prompt.replace("{{BLOG_TITLE}}", blog_title)

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[
            {"role": "assistant", "content": blog_title_prompt},
            {"role": "system", "content": blog_title},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content, prompt


if __name__ == "__main__":
    blog_title = "Dedication | Being an Entrepreneur"
    blog_body, _ = generate_content(blog_title)
    print(f"Blog Title: {blog_title}")
    print(f"Blog Body: {blog_body}")
