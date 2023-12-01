import glob
from pathlib import Path
from typing import List

import click
import openai
import os
import rich
from binaryornot.check import is_binary
from langchain.docstore.document import Document
from langchain.document_loaders.directory import _is_visible, logger
import sys

# Define ANSI escape codes for console styling
ANSI_BOLD = "\033[1m"
ANSI_GREEN = "\033[32m"
ANSI_RESET = "\033[0m"

openai.api_key = os.getenv("OPENAI_API_KEY")

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI


from langchain.chains import LLMChain

# Run the chain only specifying the input variable.

templ = """
Please help me with a question: {question}
Here is additional relevant information to answer well and correctly: {context}
"""
templ_no_context = """
Please help me with a question: {question}
"""

from langchain.document_loaders import DirectoryLoader, TextLoader

from langchain.indexes import VectorstoreIndexCreator

from rich.table import Column, Table
from rich.console import Console
from rich.pretty import Pretty
from collections import OrderedDict
import os

temp = 0.3678794412
class SpecificFilesLoader(DirectoryLoader):
    """Loading logic for loading documents from a directory."""

    def __init__(self, specific_files, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.specific_files = specific_files

    def load(self) -> List[Document]:
        """Load documents."""
        p = Path(self.path)
        docs = []
        items = self.specific_files
        for i in items:
            if i.is_file():
                if _is_visible(i.relative_to(p)) or self.load_hidden:
                    try:
                        sub_docs = self.loader_cls(str(i)).load()
                        docs.extend(sub_docs)
                    except Exception as e:
                        if self.silent_errors:
                            logger.warning(e)
                        else:
                            raise e
        return docs

@click.command()
@click.argument('question', default="")
@click.option('--context', is_flag=False, flag_value="", default="")
@click.option('--exclude_glob',is_flag=False, flag_value="", default="")
def answer(question, context, exclude_glob):
    if question == "":
        question = input("question: ")
    llm = OpenAI(temperature=temp)
    possible_files = glob.glob('**/*.*', recursive=True)
    non_binaryfiles = filter(lambda x: not is_binary(x), possible_files)
    non_binaryfiles = [x for x in non_binaryfiles if x not in glob.glob(exclude_glob, recursive=True)]
    # binaryfiles = filter(lambda x: is_binary(x), possible_files)
    nonbinaryfiles = [Path(x) for x in non_binaryfiles]
    rich.print(f"nonbinaryfiles: {nonbinaryfiles}")
    # rich.print(f"binaryfiles: {binaryfiles}")



    loader = SpecificFilesLoader(nonbinaryfiles, '.', loader_cls=TextLoader)
    index = VectorstoreIndexCreator().from_loaders([loader])
    if context:
        human_message_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=templ,
                input_variables=["question", "context"],
            )
        )
    else:
        human_message_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=templ_no_context,
                input_variables=["question"],
            )
        )
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
    chain = LLMChain(llm=llm, prompt=chat_prompt_template)
    if context:
        query = chain.run(question=question, context=context)
        q = OrderedDict([('question', question), ('context', context)])
    else:
        query = chain.run(question=question)
        q = OrderedDict([('question', question)])

    answer = index.query_with_sources(query)
    ordered_answer = OrderedDict()
    for key in q:
        if key in answer:
            ordered_answer[key] = answer[key]
    q['sources'] = answer['sources']
    ordered_answer['sources'] = answer['sources']
    print_qa(q, ordered_answer)


def print_qa(q, a):
    # Create a table and add the header rows
    table = Table(title="qq")
    for key in ["", *list(q.keys())]:
        table.add_column(key, justify="left")

    # Add the values to the table
    table.add_row("q", *[Pretty(value) for value in q.values()])
    table.add_row("a", *[Pretty(value) for value in a.values()])
    console = Console()
    console.print(table)


if __name__ == '__main__':
    if sys.argc < 2:
        answer()
    else:
        answer(sys.argv[1])
