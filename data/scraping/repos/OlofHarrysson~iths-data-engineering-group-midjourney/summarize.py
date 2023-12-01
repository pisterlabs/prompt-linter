import argparse
import json
import os  # for loading api from .env file
from pathlib import Path
from time import monotonic  # Times the run time of the chain

import openai
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI  # for generating the summary
from langchain.docstore.document import Document  # for storing the text
from langchain.prompts import PromptTemplate  # Template for the prompt

from newsfeed import utils
from newsfeed.datatypes import BlogInfo, BlogSummary
from newsfeed.hugging_face_model import summarize_text_with_hugging_face


# Read blog_data from DataWearhouse into a list of articles
def get_articles_from_folder(blog_name):
    # define path to articles in DataWarehouse
    path_articles = Path("data/data_warehouse") / blog_name / "articles"

    # Check if directory exists otherwise return None
    if not path_articles.exists():
        raise FileNotFoundError(f"Directory {path_articles} does not exist.")

    # create a list with all articles which are .json
    articles_list = [article for article in path_articles.iterdir() if article.suffix == ".json"]

    # read in content of all articles formated according to BlogInfo model into a list
    articles_all = []
    for article in articles_list:
        with open(article) as f:
            dict_repr_of_json = json.load(
                f
            )  # json.load() directly loads JSON content into a dictionary
            parsed_article = BlogInfo.parse_obj(
                dict_repr_of_json
            )  # Use parse_obj() for dict input (deserialization)
            articles_all.append(parsed_article)
    return articles_all


def get_summaries_from_folder(blog_name):
    path_summaries = Path("data/data_warehouse") / blog_name / "summaries"
    summaries_list = [summary for summary in path_summaries.iterdir() if summary.suffix == ".json"]

    summaries = []
    for summary in summaries_list:
        with open(summary) as f:
            dict_repr_of_json = json.load(f)
            parsed_summary = BlogSummary.parse_obj(dict_repr_of_json)
            summaries.append(parsed_summary)
    return summaries


# Set up OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Summarization function
def summarize_text(blog_text, prompt_template):
    model_name = "gpt-3.5-turbo"

    # Converts each part into a Document object
    docs = [Document(page_content=blog_text)]

    # Loads the lanugage model
    llm = ChatOpenAI(temperature=0, openai_api_key=openai.api_key, model_name=model_name)

    # Defines prompt template
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    # Define model parameters
    verbose = False  # If set to True, prints entire un-summarized text

    # Loads appropriate chain based on the number of tokens. Stuff or Map Reduce is chosen
    chain = load_summarize_chain(
        llm,
        chain_type="stuff",
        prompt=prompt,
        verbose=verbose,
    )
    summary = chain.run(docs)

    return summary


# Takes an article represented by a BlogInfo instance, generates a summary and constructs a new BlogSummary instance
def transform_to_summary(article: BlogInfo, model_type) -> BlogSummary:
    # Call the summarize_text function twice to generate both summaries
    technical_prompt = "Write a very short, concise technical summary of the following text. Not more than 600 characteres:{text}"
    non_technical_prompt = "Write a very short, concise non-technical summary suitable for a general audience of the following text. Not more than 600 characteres:{text}"

    if model_type == "api":
        technical_summary = summarize_text(article.blog_text, technical_prompt)
        non_technical_summary = summarize_text(article.blog_text, non_technical_prompt)
    else:
        technical_summary = summarize_text_with_hugging_face(article.description)
        non_technical_summary = summarize_text_with_hugging_face(article.description)

    return BlogSummary(
        unique_id=article.unique_id,
        title=article.title,
        blog_summary_technical=technical_summary,
        blog_summary_non_technical=non_technical_summary,
    )


# save all summeries to DataWarehouse
def save_blog_summaries(articles, blog_name, model_type):
    # Define path to summary folder in data warehouse and create it if does not exist
    path_summaries = Path("data/data_warehouse") / blog_name / "summaries"
    path_summaries.mkdir(exist_ok=True, parents=True)

    for article in articles:
        print(article.get_filename())
        # generate summary for current article
        summary = transform_to_summary(article, model_type)

        # define path where summary will be saved
        save_path = path_summaries / summary.get_filename()

        with open(save_path, "w+") as f:
            f.write(
                summary.json(indent=2)
            )  # Serialize BlogSummary instance to JSON and write it to the file


def main(blog_name, model_type):
    path_summaries = Path("data/data_warehouse") / blog_name / "summaries"
    if path_summaries.exists():
        print("Starting the summarization process...")
        print(f"Summary path exists for blog: {blog_name}")
        # Retrieve a lists of articles and summaries from the specified blog folders
        path_summaries.mkdir(parents=True, exist_ok=True)
        print("Fetching existing articles...")
        articles_all = get_articles_from_folder(blog_name)
        print("Fetching existing summaries...")
        summaries = get_summaries_from_folder(blog_name)
        print("Filtering articles that need summarization...")
        # Generate list of articles which are present in articles_all but not in summaries based on unique_id
        summaries_unique_ids_list = [summary.unique_id for summary in summaries]
        articles = [
            article
            for article in articles_all
            if article.unique_id not in summaries_unique_ids_list
        ]
    else:
        print(f"Summary path does not exist for blog: {blog_name}. Creating one...")
        print("Fetching all articles for summarization...")
        articles = get_articles_from_folder(blog_name)
        # Save summaries for the retrieved articles to the Data Warehouse
    save_blog_summaries(articles, blog_name, model_type)
    print("Summarization completed.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--blog_name", type=str, required=True, choices=["mit", "google_ai", "ai_blog", "open_ai"]
    )
    parser.add_argument("--model_type", type=str, default="api", choices=["api", "local_model"])

    # print("Starting the summarization process...")
    # print("Summarization completed.")
    return parser.parse_args()


# Check if the script is run directly
if __name__ == "__main__":
    # Parse command-line arguments using pare_args() from utils
    args = parse_args()  # args = Namespace(blog_name='mit') if program ran with --blog_space mit
    # Extract the blog_name from the parsed arguments
    blog_name = args.blog_name
    model_type = args.model_type
    main(blog_name, model_type)
