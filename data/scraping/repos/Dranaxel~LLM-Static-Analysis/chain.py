import os
import json
import sys
import asyncio
from rich.table import Table
from rich.console import Console

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from loguru import logger

logger.remove()
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logger.add(sink=sys.stderr, level=LOGLEVEL)


async def review_code(code, filename):
    logger.debug(f"reviewing {filename}")
    template = PromptTemplate.from_template(
        """
     Please review the following code according to the criteria outlined below. Your review should be returned as a JSON objects. Each JSON object should contain the following keys:

    2. "filename": the name of the file
    2. "codeQuality": A score out of ten that represents the overall quality of the code. Please consider factors such as readability, efficiency, and adherence to best practices when determining this score.
    3. "goodPoints": An array of points that highlight the strengths of the code. This could include things like effective use of data structures, good commenting, efficient algorithms, etc.
    4. "badPoints": An array of points that highlight areas where the code could be improved. This could include things like unnecessary repetition, lack of comments, inefficient algorithms, etc.

    Please ensure that your review is thorough and constructive. Remember, the goal is to help the coder improve, not to criticize them unnecessarily.

    Example of expected output:
        {{
            "filename": {{filename}},
            "codeQuality": 5,
            "goodPoints": ["Efficient algorithms"],
            "badPoints": ["Lack of comments", "Inefficient data structures"]
        }}

    output only the json

    Filename: {filename}
    Code: {code}

    """
    )
    llm = ChatOpenAI(temperature=0.8, model="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=template)
    review = await chain.arun(code=code, filename=filename)
    logger.debug(review)
    return review


def load_code_from_file(filename):
    with open(filename, "r") as file:
        return file.read()


async def review_directory(directory, extension=None):
    reviews = []
    for root, dirs, files in os.walk(directory):
        logger.debug(files)
        for file in files:
            if extension is None or file.endswith(extension):
                filename = os.path.join(root, file)
                code = load_code_from_file(filename)
                reviews.append(review_code(code, filename))
    reviews = await asyncio.gather(*reviews)
    reviews = [json.loads(_) for _ in reviews]
    return reviews


def display_reviews(reviews, sort_by="filename"):
    reviews.sort(key=lambda review: review[sort_by])
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("Filename")
    table.add_column("Code Quality")
    table.add_column("Good Points", justify="full")
    table.add_column("Bad Points", justify="full")

    for review in reviews:
        color = (
            "green"
            if review["codeQuality"] > 9
            else "bright_green"
            if review["codeQuality"] > 7
            else "yellow"
            if review["codeQuality"] > 5
            else "orange"
            if review["codeQuality"] > 3
            else "red"
        )
        table.add_row(
            str(review["filename"]),
            str(review["codeQuality"]),
            str("\n".join(review["goodPoints"])),
            str("\n".join(review["badPoints"])),
            style=color,
        )

    console = Console()
    console.print(table)
