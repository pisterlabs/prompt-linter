import os
import tempfile
import openai
import fileinput
import shutil
import zipfile
from halo import Halo
import re
import ast

import config

openai.api_key = config.OPENAI_API_KEY

spinner = Halo(text="Loading", spinner="dots")

if os.path.exists("blog_gen/website_serve"):
    shutil.rmtree("blog_gen/website_serve")

source_dir = r"blog_gen/websiteCopy"
destination_dir = r"blog_gen/website_serve"
shutil.copytree(source_dir, destination_dir)



def create_blog_posts(num_posts=3, topic="Tea", author="Umar Dabhoiwala"):
    """
    Create a specified number of blog posts on a given topic using GPT-3.5-turbo and save them as Markdown files.

    :param num_posts: Number of blog posts to create (default: 3)
    :param topic: Topic of the blog posts (default: "Tea")
    :param author: Author of the blog posts (default: "Umar Dabhoiwala")
    """

    init_prompt = """
    You are a blog creation helper, given a theme return a blog post in the following format

    Title: My super title
    Date: 2010-12-03 10:20
    Modified: 2010-12-05 19:30
    Keywords: XXX, XXX, XXX
    Category: Python
    Tags: pelican, publishing
    Slug: my-super-post
    Authors: Umar Dabhoiwala
    Summary: Summary of Content

    This is the content of my super blog post.
    """

    messages_cust = [{"role": "system", "content": init_prompt}]

    for i in range(num_posts):

        spinner.start(f"Generating Blog Post {i+1}/{num_posts}")

        prompt = f"""Write a blog post about a niche topic relating to {topic} be creative but keep it concise less than 200 words,
        in the Author Field put {author}, replace all XXX's in the metadata with the appropriate tags,
        and replace the content with the content of the blog post."""
        messages_cust.append({"role": "user", "content": prompt})

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages_cust,
        )

        spinner.succeed(f"Blog Post {i+1}/{num_posts} Generated")

        response = completion['choices'][0]['message']['content']
        messages_cust.append({"role": "system", "content": response})

        save_response_as_markdown(response)


def save_response_as_markdown(response):
    """
    Save a GPT-3 generated response as a Markdown file in the 'website_serve/content' directory.

    :param response: GPT-3 generated response containing blog post metadata and content
    """
    lines = response.split('\n')
    slug, summary = None, None

    for line in lines:
        if line.startswith('Slug:'):
            slug = line.split(': ')[1].strip()
        if line.startswith('Summary:'):
            summary = line.split(': ')[1].strip()

    directory = "blog_gen/website_serve/content"
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = f"{slug}.md"
    filepath = os.path.join(directory, filename)

    with open(filepath, "w") as file:
        line_count = 0
        for line in response.splitlines():
            line_count += 1
            if line.strip():
                file.write(line + "\n")
            elif line_count > 16:
                file.write(line + "\n")

def zip_directory(dir_path, zip_path):
    """
    Creates a zip archive of a directory.

    :param dir_path: The path of the directory to zip.
    :param zip_path: The path of the zip file to create.
    """
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(file_path, dir_path))


def update_file(file_path, new_values):
    with open(file_path, 'r') as file:
        content = file.read()

    for key, value in new_values.items():
        old_value_pattern = f"{key} = '.*?'"
        new_value = f"{key} = '{value}'"
        content = re.sub(old_value_pattern, new_value, content)

    with open(file_path, 'w') as file:
        file.write(content)

def new_value_gen(f_topic, f_author):

    backup = {
        'AUTHOR': f_author,
        'SITENAME': f_topic + ' Blog',
        'SITETITLE': f_topic + ' Blog',
        'SITESUBTITLE': "All you need to know about " + f_topic,
        'SITEDESCRIPTION': "All you need to know about " + f_topic
    }

    new_values = {
        'AUTHOR': "XXX",
        'SITENAME': "XXX",
        'SITETITLE': "XXX",
        'SITESUBTITLE': "XXX",
        'SITEDESCRIPTION': "XXX"
    }


    prompt = f"Write values into the following object {new_values}. For context the topic is {f_topic} and the author is {f_author}. Write something for each field"

    mess = [{"role": "user", "content": prompt}]

    completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages= mess,
    )

    response = completion['choices'][0]['message']['content']


    return response, backup


def fixFile(f_topic, f_author):

    new_values, backup = new_value_gen(f_topic, f_author)

    try:
        thing = ast.literal_eval(new_values)
        update_file("blog_gen/website_serve/pelicanconf.py", thing)
    except Exception as e:
        print(e)
        print("no worries")
        update_file("blog_gen/website_serve/pelicanconf.py", backup)

def fullrun(f_num_posts, f_topic, f_author):
    if os.path.exists("blog_gen/website_serve"):
        shutil.rmtree("blog_gen/website_serve")

    source_dir = r"blog_gen/websiteCopy"
    destination_dir = r"blog_gen/website_serve"
    shutil.copytree(source_dir, destination_dir)


    create_blog_posts(num_posts=f_num_posts, topic=f_topic, author= f_author)

    fixFile(f_topic, f_author)

    zip_directory('blog_gen/website_serve', 'blog_gen/website_serve.zip')

