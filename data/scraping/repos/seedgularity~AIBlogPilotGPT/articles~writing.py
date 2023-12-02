import os
import sys
import time
import requests
import openai
from serpapi import GoogleSearch
from dotenv import load_dotenv
from termcolor import colored
from tqdm import tqdm
import colorama
colorama.init(autoreset=True)
from config import Config

load_dotenv()

cfg = Config()

# Configure OpenAI API key
try:
    openai.api_key = cfg.openai_api_key
    browserless_api_key = cfg.browserless_api_key
    llm_model = cfg.llm_model
except KeyError:
    sys.stderr.write("OpenAI key configuration failed.")
    exit(1)


def find_conclusion(text, function_to_call):
    if "In conclusion," in text:
        print("Found In conclusion text. Removing...")
        return function_to_call(text)
    return text
def in_conclusion_killer(text):
    ###THIS BLOODY THING###
    prompt = f"""Remove in conclusion text. Don't add anything.: {text}"""
    chunked_output = ""
    for chunk in openai.ChatCompletion.create(
            model=cfg.llm_model,
            temperature=1,
            stream=True,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'system', 'content': "You're an expert in blogging, research and SEO."},
                {'role': 'system', 'content': 'Your name is BloggingGPT.'},
                {'role': 'system', 'content': 'Content you write is well SEO Optimised.'},
                {'role': 'system', 'content': 'You use engaging tone of voice.'},
                {'role': 'system', 'content': 'You strictly follow user request.'},
                {'role': 'system', 'content': 'You strictly return content user asked for only.'},
                {'role': 'system', 'content': "You don't say or add anything, just return the content."},
                {"role": "user", "content": prompt}
            ]
    ):
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            # print(content, end='')
            chunked_output += content

    return chunked_output
def write_intro(title, toneofvoice):

    prompt = f"""
As an AI blog post writer, your task is to craft an engaging and professional introduction paragraph for an article. Here are the details you need to consider:

The article's title is {title}. However, do not include the title in the introduction.
Write in the style of a professional blogger crafting a long-form article.
The article will have following tone of voice: {toneofvoice}.
Do not include any form of concluding statements like 'in conclusion'.
Remember, your goal is to create an introduction that hooks the reader and sets the stage for the rest of the article."""

    chunked_output = ""
    for chunk in openai.ChatCompletion.create(
            model=cfg.llm_model,
            temperature=1,
            stream=True,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'system', 'content': "You're an expert in blogging, research and SEO."},
                {'role': 'system', 'content': 'Your name is BloggingGPT.'},
                {'role': 'system', 'content': 'Content you write is well SEO Optimised.'},
                {'role': 'system', 'content': 'You use engaging tone of voice.'},
                {"role": "user", "content": prompt}
            ]
    ):
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            #print(content, end='')
            chunked_output += content
    chunked_output = find_conclusion(chunked_output, in_conclusion_killer)
    return chunked_output


def write_section(title, article_description, heading, heading_description, toneofvoice):
    if article_description == "":
        article_description = "Not available, use the title"
    if heading_description == "":
        heading_description = "Not available, use the heading"

    prompt = f"""
As an AI blog post section writer, your task is to generate unique, compelling, and SEO-optimized content for various blog post sections. Here are the details you need to consider:
You will not include any concluding summaries.
You will not include section headings.
The article will have following tone of voice: {toneofvoice}.
You will be provided with an article title {title}, an article description {article_description}, a section heading {heading}, and a section description {heading_description}.
Using these inputs, generate captivating, grammatically correct, and easy-to-read content that is suitable for the respective section.
Make important parts of the text bold - but focus on keywords the article is targeting. You can also use quotes and citations.
The content should engage readers and facilitate their understanding of the blog post's content. Maintain an engaging tone of voice throughout.
The content should be ready to be copied and pasted directly into Wordpress with Gutenberg formatting, without the need for any additional formatting.
Remember, your goal is to create a section body that aligns with the provided inputs and is optimized for search engines."""

    chunked_output = ""
    for chunk in openai.ChatCompletion.create(
            model=cfg.llm_model,
            temperature=1,
            stream=True,
            messages=[
                {'role': 'system', 'content': 'Your name is BloggingGPT.'},
                {'role': 'system', 'content': "You're an expert in blogging, research and SEO."},
                {'role': 'system', 'content': 'Content produced is well SEO Optimised.'},
                {'role': 'system', 'content': 'You strictly return content user asked for only.'},
                {"role": "user", "content": prompt}
            ]
    ):
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            #print(content, end='')
            chunked_output += content

    chunked_output = in_conclusion_killer(chunked_output)
    return chunked_output

def write_subsection(title, article_description, heading, heading_description, subheading, subheading_description, toneofvoice):
    if article_description == "":
        article_description = "Not available, use the title"
    if heading_description == "":
        heading_description = "Not available, use the heading"
    if subheading_description == "":
        subheading_description = "Not available, use the subheading"

    prompt = f"""As an AI blog post section writer, your task is to generate unique, compelling, and SEO-optimized content for various blog post sections and subsections. Here are the details you need to consider:

You will not include any concluding summaries.
You will not include section or subsections headings.
The article will have following tone of voice: {toneofvoice}.
You will be provided with an article title {title}, an article description {article_description}, a section heading {heading}, a section description {heading_description}, a subsection heading {subheading}, and a subsection description {subheading_description}.
Using these inputs, generate captivating, grammatically correct, and easy-to-read content that is suitable for the respective section and subsection.
Make important parts of the text bold. You can also use quotes and citations. Add bulleted lists.
When relevant, add pros and cons comparisons.
The content should engage readers and facilitate their understanding of the blog post's content. Maintain an engaging tone of voice throughout.
The content should be ready to be copied and pasted directly into Wordpress with Gutenberg formatting, without the need for any additional formatting.
Remember, your goal is to create a section body and subsection body that align with the provided inputs and are optimized for search engines."""

    chunked_output = ""
    for chunk in openai.ChatCompletion.create(
            model=cfg.llm_model,
            temperature=1,
            stream=True,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'system', 'content': "You're an expert in blogging, research and SEO."},
                {'role': 'system', 'content': 'Your name is BloggingGPT.'},
                {'role': 'system', 'content': 'Content produced is well SEO Optimised.'},
                {'role': 'system', 'content': 'You use engaging tone of voice.'},
                {'role': 'system', 'content': 'You strictly return content user asked for only.'},
                {"role": "user", "content": prompt}
            ]
    ):
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            #print(content, end='')
            chunked_output += content

    chunked_output = find_conclusion(chunked_output, in_conclusion_killer)
    return chunked_output

if __name__ == "__main__":
    write_section()