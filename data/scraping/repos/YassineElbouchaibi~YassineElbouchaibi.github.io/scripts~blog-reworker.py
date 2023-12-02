import datetime
import json
import os
from ghapi.all import GhApi
import openai

AI_MODEL = "gpt-3.5-turbo"

# Instructions for the AI
def get_system_prompt():
    return f"""
You are a dev blog author who wrote an article and will now receive feedback about it. When given the blog article you previously wrote and feedback about it, you will make sure to correct your article to fix every element mentioned in the feedback. The will answer back with the fixed blog article in markdown language for me to publish on my dev blog. Make sure the include the entire article in your answer, even the parts that were correct. Feel free to bring extra changes that will benefit the article. The blog article should have a reading time of around 15 minutes. Be very technical, Add an actual example and don't hesitate to make use of code blocks. Use rust, python or typescript for all code examples. The output should start with a header in the following format:
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
"""

def get_user_prompt(previous_article_content: str, feedback: str):
    return f"Here is the article you previously wrote:\n\n{previous_article_content}\n\nHere is the feedback you received:\n\n{feedback}\n\nPlease fix the article to address the feedback and make it better."

# Verify ENV vars are set
OUT_PATH = os.getenv("OUT_PATH")
if OUT_PATH is None:
    raise Exception("OUT_PATH is not set")

openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise Exception("OPENAI_API_KEY is not set")

if os.getenv("GITHUB_TOKEN") is None:
    raise Exception("GITHUB_TOKEN is not set")

GITHUB_EVENT_PATH = os.getenv("GITHUB_EVENT_PATH")
if GITHUB_EVENT_PATH is None:
    raise Exception("GITHUB_EVENT_PATH is not set")

# Load GitHub event
with open(GITHUB_EVENT_PATH, "r") as f:
    github_event = json.load(f)

# Verify the event is a comment
if github_event["action"] != "created" or github_event.get("comment") is None:
    print("Event is not a comment, skipping...")
    exit(0)

# Comment author must be an admin
if github_event["comment"]["author_association"] != "OWNER":
    print("Comment author is not an admin, skipping...")
    exit(0)

# Comment must start with @bot or @actions-user
if not github_event["comment"]["body"].startswith("@bot") and not github_event["comment"]["body"].startswith("@actions-user"):
    print("Comment does not start with @bot or @actions-user, skipping...")
    exit(0)

pr_number = github_event["issue"]["number"]
at_user = "@actions-user" if github_event["comment"]["body"].startswith("@actions-user") else "@bot"

# Initialize GitHub API
gh_api = GhApi()

# Get the previous article content
files = gh_api.pulls.list_files(
    owner="YassineElbouchaibi",
    repo="YassineElbouchaibi.github.io",
    pull_number=github_event["issue"]["number"]
)
files = [file for file in files if file.filename.startswith("content/blog/")]
files = [file for file in files if file.filename.endswith(".md")]
if len(files) == 0:
    print("No markdown files found in pull request, skipping...")
    exit(0)

if len(files) > 1:
    print(f"Multiple markdown files [{', '.join(files)}] found in pull request, skipping...")
    exit(0)

# Checkout to PR branch
os.system(f"git fetch origin pull/{pr_number}/head:pr-{pr_number}")
os.system(f"git checkout pr-{pr_number}")

previous_file = files[0].filename

if not os.path.exists(previous_file):
    print(f"File {previous_file} not found on disk, skipping...")
    exit(0)

with open(previous_file, "r") as f:
    previous_article_content = f.read()

# Fetch and checkout to original PR branch
article_slug = previous_article_content.split("slug:")[1].split("\n")[0].strip()
os.system(f"git fetch origin content/{article_slug}")
os.system(f"git checkout content/{article_slug}")

# Get the feedback - The comment must be in the following format:
# <at_user>
# <feedback>
# e.g.
# @bot
# - The title is too long
# - The description is too short
# - The article is too technical
feedback = github_event["comment"]["body"].split(at_user)[1].strip()

# Get ai prompt
system_prompt = get_system_prompt()
user_prompt = get_user_prompt(
    previous_article_content=previous_article_content,
    feedback=feedback
)

# Create a new blog article
print("Fixing article...")
completion = openai.ChatCompletion.create(
  model=AI_MODEL,
  messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
  ]
)
print("New Article generated! Writing to file...")

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

# Remove previous article from disk and its parent directory
print(f"git rm -r {os.path.dirname(previous_file)}")
os.system(f"git rm -r {os.path.dirname(previous_file)}")

# Write the article to a file
os.makedirs(f"{OUT_PATH}/{article_slug}", exist_ok=True)
with open(f"{OUT_PATH}/{article_slug}/index.md", "w") as f:
    f.write(article_content)
print(f"Article written to file {OUT_PATH}/{article_slug}/index.md! Pushing to GitHub...")

# Add the article to git
print(f"git add {OUT_PATH}/{article_slug}")
os.system(f"git add {OUT_PATH}/{article_slug}")

safe_article_title = article_title.replace("'", "\\'")
print(f"git commit -m 'Fixed blog article: {safe_article_title}'")
os.system(f"git commit -m 'Fixed blog article: {safe_article_title}'")

print(f"git push origin content/{article_slug}")
os.system(f"git push origin content/{article_slug}")

# Checkout the dev branch
os.system("git checkout dev")

print("Done!")
