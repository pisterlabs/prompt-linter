import argparse
import json
import os
import tempfile
from pathlib import Path

import openai
from dotenv import load_dotenv
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, pipeline

load_dotenv()

from newsfeed.datatypes import BlogInfo, BlogSummary
from newsfeed.extract_articles import sanitize_filename


def summarize_text(blog_post_path, local_model=None, sum_type="tech"):
    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key
    # loading text as langchain document
    loader = TextLoader(blog_post_path)
    blog_post = loader.load()

    if local_model == "gpt2":
        # just text for local models
        blog_post_text = json.loads(blog_post[0].page_content)["blog_text"]

        checkpoint = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)  # this selects GPT2TokenizerFast

        # max token length is 1024
        # limit input tokens to allow enough space for generation
        tokens_text = tokenizer.tokenize(blog_post_text)
        tokens_tldr = tokenizer.tokenize("\n\nTL;DR:\n")
        tokens = tokens_text[:800] + tokens_tldr

        # convert back to string for pipeline
        query = tokenizer.convert_tokens_to_string(tokens)

        # pipeline uses tokenizer GPT2TokenizerFast and model GPT2LMHeadModel
        pipe = pipeline(
            "text-generation",
            model=checkpoint,
            max_new_tokens=(1024 - len(tokens)),
            temperature=1.5,  # higher value softens output distribution (reducing confidence in predictions, adding more randomness)
        )
        pipe_out = pipe(query)
        summary = pipe_out[0]["generated_text"][len(query) :]

    elif local_model == "t5":
        # just text for local models
        blog_post_text = json.loads(blog_post[0].page_content)["blog_text"]

        checkpoint = "t5-small"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)  # T5TokenizerFast

        # maximum sequence length for model is 512
        tokens = tokenizer.tokenize(blog_post_text, max_length=512, truncation=True)
        tokens = tokens[:509]  # for some reason pipeline sees 515 tokens
        query = tokenizer.convert_tokens_to_string(tokens)

        # pipeline uses tokenizer T5TokenizerFast and model T5ForConditionalGeneration
        pipe = pipeline("summarization", model=checkpoint, temperature=1.5, do_sample=True)
        pipe_out = pipe(query, max_length=257)
        summary = pipe_out[0]["summary_text"]

    elif local_model == None:
        prompt = select_prompt(sum_type)

        # define what LLM to use
        # using OpenAI chat LLM API requires env variable OPEN_AI_KEY to be set with API key
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

        # create a "chain" object for running queries against a specified LLM, with customizable prompt
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # send document to the llm_chain
        # StuffDocumentsChain "stuffs" list of documents + prompt into single call
        # MapReduceChain splits it into several combines and then combines output
        # document_variable_name can be used to customize where in the prompt the documents are inserted?
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        summary = stuff_chain.run(blog_post)

    else:
        raise NotImplementedError(
            f"Local model {local_model} not implemented. Please choose between 'gpt2' and 't5'"
        )

    return summary


def select_prompt(sum_type):
    if sum_type == "tech":
        prompt_template = """Write a concise summary for an AI researcher of the following:
        "{text}"
        CONCISE SUMMARY:"""
    elif sum_type == "ntech":
        prompt_template = """Write a very simple summary, suitable for children, of the following:
        "{text}"
        SIMPLE SUMMARY:"""
    elif sum_type == "swe":
        prompt_template = """Skriv en sammanfattning på svenska av följande text:
        "{text}"
        Sammanfattning:"""
    else:
        raise NotImplementedError(
            f"Summarization type {sum_type} not implemented. Please choose 'tech', 'ntech', or 'swe'"
        )

    prompt = PromptTemplate.from_template(prompt_template)
    return prompt


# main takes the argument --source aws, or --source mit
def main(source, local_model=None, sum_type="tech"):
    path_article_dir = Path(f"data/data_warehouse/{source}/articles")
    path_summary_dir = Path(f"data/data_warehouse/{source}/summarized_articles")

    if local_model:
        path_summary_dir = path_summary_dir / local_model
    else:
        path_summary_dir = path_summary_dir / sum_type

    path_summary_dir.mkdir(parents=True, exist_ok=True)

    already_summarized = set(os.listdir(path_summary_dir))

    file_list = os.listdir(path_article_dir)
    new_articles = []

    for file_name in file_list:
        summary_filename_check = f"Summary_of_{file_name}"

        if summary_filename_check in already_summarized:
            print(f"Skipping already summarized article: {file_name}")
            continue

        # if new summary is generated, add to list
        new_articles.append(file_name)
        # file path for the article that's being summarized/looked at
        current_article_path = path_article_dir / file_name
        summary_text = summarize_text(current_article_path, local_model, sum_type)
        print(f"Generated summary for {file_name}")

        blog_summary = BlogSummary(
            unique_id=f"summary_{file_name}",
            title=f"Summary of {Path(file_name).stem}",
            text=summary_text,
        )
        # gets the name of the article
        summary_filename = blog_summary.get_filename()
        summary_filename = sanitize_filename(summary_filename)

        with open(path_summary_dir / summary_filename, "w") as f:
            f.write(blog_summary.json())
        already_summarized.add(summary_filename)

    return new_articles


blog_names = ["mit", "aws", "openai"]


# run python summarize.py --source mit OR aws
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        choices=blog_names,
        default="mit",
        help=f"Blog source to summarize, allowed arguments are: {blog_names}.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["t5", "gpt2"],
        # default="",
        help=f"Use local model for summarization",
    )
    parser.add_argument(
        "--stype",
        type=str,
        choices=["tech", "ntech", "swe"],
        default="tech",
        help=f"What type of summarization do you want (techincal 'tech', non-technical 'ntech', or swedish 'swe')",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(source=args.source, local_model=args.model, sum_type=args.stype)
