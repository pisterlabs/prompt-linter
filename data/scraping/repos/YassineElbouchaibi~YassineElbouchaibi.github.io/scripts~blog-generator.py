import datetime
import os
from ghapi.all import GhApi
import openai

AI_MODEL = "gpt-3.5-turbo"

# Instructions for the AI
def get_system_prompt(existing_articles: list = []):
    existing_articles_join_string = '\n    - '
    return f"""
You are a dev blog author. When given a blog title you will write a detailed blog article in markdown language for me to publish on my dev blog. When given guidelines make use of them while generating the article. If you are not given blog title, select a random subject that relates to dev and tech and write about it. Don't hesitate to tweak the title to make it better. The blog article should have a reading time of around 15 minutes. Be very technical, Add an actual example and don't hesitate to make use of code blocks.  Use rust, python or typescript for all code examples. The output should start with a header in the following format:
---
title: 
slug:
date: "2023-<current-month>-<current-day>T22:12:03.284Z"
tags:
description:
author: {AI_MODEL.upper()}
---
e.g.
---
title: The Benefits and Drawbacks of Different Programming Languages and Which Ones Are Right for Certain Projects
date: "2023-03-09T22:12:03.284Z"
tags: programming languages, coding, software development, tech
description: "In this blog post, we explore the various benefits and drawbacks of popular programming languages and which ones are suitable for specific types of projects. We cover languages such as Java, Python, C++, JavaScript, and more, providing insight into their strengths and weaknesses. By the end of this post, readers will have a better understanding of which language is the best fit for their next coding project."
author: {AI_MODEL.upper()}
---
Do not write a blog article if my dev blog already covers the subject. Here is a list of the articles already available on my website:
    - {existing_articles_join_string.join(existing_articles)}
"""

def get_user_prompt():
    return "Write a new dev blog article."

# Verify ENV vars are set
OUT_PATH = os.getenv("OUT_PATH")
if OUT_PATH is None:
    raise Exception("OUT_PATH is not set")

openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise Exception("OPENAI_API_KEY is not set")

if os.getenv("GITHUB_TOKEN") is None:
    raise Exception("GITHUB_TOKEN is not set")

# Initialize GitHub API
gh_api = GhApi()

# Get list of articles in review (Open PRs)
articles_in_review = []
for pr in gh_api.pulls.list(
    owner="YassineElbouchaibi",
    repo="YassineElbouchaibi.github.io",
    state="open"
):
    if pr.head.ref.startswith("content/"):
        articles_in_review.append(pr.head.ref.split("/")[1])

# Augment system_prompt with existing articles
existing_articles = []
for root, dirs, files in os.walk(OUT_PATH):
    for file in files:
        if file == "index.md":
            with open(os.path.join(root, file), "r") as f:
                article_content = f.read()
                article_title = article_content.split("title:")[1].split("\n")[0].strip()
                existing_articles.append(article_title)

# Get ai prompt
system_prompt = get_system_prompt(
    existing_articles=existing_articles + articles_in_review
)
user_prompt = get_user_prompt()

# Create a new blog article
print("Generating new blog article...")
completion = openai.ChatCompletion.create(
  model=AI_MODEL,
  messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
  ]
)
print("Article generated! Writing to file...")

# Extract the article content and slug
article_content = completion.choices[0].message.content
article_title = article_content.split("title:")[1].split("\n")[0].strip()
article_slug = article_content.split("slug:")[1].split("\n")[0].strip()

# Replace date in article content
article_content = article_content.replace(
    article_content.split("date:")[1].split("\n")[0].strip(),
    # Set the date to the current date
    f"\"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z\""
)

# Write the article to a file
os.makedirs(f"{OUT_PATH}/{article_slug}", exist_ok=True)
with open(f"{OUT_PATH}/{article_slug}/index.md", "w") as f:
    f.write(article_content)
print(f"Article written to file {OUT_PATH}/{article_slug}/index.md! Pushing to GitHub...")

# Create a new branch and commit the new article
print(f"git checkout -b content/{article_slug}")
os.system(f"git checkout -b content/{article_slug}")

print(f"git add {OUT_PATH}/{article_slug}")
os.system(f"git add {OUT_PATH}/{article_slug}")

safe_article_title = article_title.replace("'", "\\'")
print(f"git commit -m 'Add new blog article: {safe_article_title}'")
os.system(f"git commit -m 'Add new blog article: {safe_article_title}'")

print(f"git push origin content/{article_slug}")
os.system(f"git push origin content/{article_slug}")

# Create a pull request
print("Creating pull request...")
gh_api.pulls.create(
    owner="YassineElbouchaibi",
    repo="YassineElbouchaibi.github.io",
    title=f"Add new blog article: {article_title}",
    body=f"This pull request adds a new blog article: {article_title}",
    head=f"content/{article_slug}",
    base="dev"
)
print("Pull request created!")

# Checkout the dev branch
os.system("git checkout dev")

print("Done!")
