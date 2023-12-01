import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from operator import itemgetter
from pathlib import Path
from typing import Dict

import click
import nbformat
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import format_document
from langchain.schema.messages import SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from nbformat.v4 import (
    new_code_cell,
    new_markdown_cell,
    new_notebook,
    reads,
    writes,
)

from nbwrite.config import Config
from nbwrite.constants import TEMPLATE_STRING
from nbwrite.index import create_index

logger = logging.getLogger(__name__)


def get_llm(**llm_kwargs: Dict[str, any]):
    return ChatOpenAI(**llm_kwargs)


def gen(
    config: Config,
):
    now = datetime.now()

    if os.getenv("NBWRITE_PHOENIX_TRACE"):
        click.echo("Enabling Phoenix Trace")
        try:
            from phoenix.trace.langchain import (
                LangChainInstrumentor,
                OpenInferenceTracer,
            )

            tracer = OpenInferenceTracer()
            LangChainInstrumentor(tracer).instrument()
        except ModuleNotFoundError:
            click.echo(
                "In order to use Phoenix Tracing you must `pip install 'nbwrite[tracing]'"
            )
            exit(1)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=config.generation.system_prompt),
            HumanMessagePromptTemplate.from_template(TEMPLATE_STRING),
        ]
    )

    if len(config.packages) > 0:
        try:
            retriever = create_index(
                config.packages,
                config.generation.retriever_kwargs,
                config.generation.text_splitter_kwargs,
            )
            context_chain = itemgetter("task") | retriever | _combine_documents
        except ModuleNotFoundError:
            click.echo(
                "In order to use `packages`, you must `pip install 'nbwrite[rag]'`"
            )
            exit(1)
    else:
        context_chain = RunnableLambda(lambda _: "none")

    llm = get_llm(**config.generation.llm_kwargs) | StrOutputParser()
    chain = (
        {
            "context": context_chain,
            "task": itemgetter("task"),
            "steps": itemgetter("steps"),
            "packages": itemgetter("packages"),
        }
        | prompt
        | RunnableParallel(**{str(gg): llm for gg in range(config.generation.count)})
    )

    click.echo(f"Invoking LLM")
    code_out = chain.invoke(
        {
            "task": config.task,
            "steps": "\n".join(config.steps),
            "packages": "\n".join(config.packages),
        }
    )

    out_dir = Path(config.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for generation in code_out.keys():
        try:
            nb = new_notebook()

            sections = re.split(r"```(?:python\n)?", code_out[generation])

            for i in range(0, len(sections)):
                if i % 2 == 0:
                    nb.cells.append(new_markdown_cell(sections[i]))
                else:
                    nb.cells.append(new_code_cell(sections[i]))

            time = now.strftime("%Y-%m-%d_%H-%M-%S")
            filename = Path(config.out) / f"{time}-gen-{generation}.ipynb"
            string = writes(nb)
            _ = reads(string)

            nbformat.write(nb, (filename.as_posix()))
            click.echo(f"Wrote notebook to {filename}")
        except Exception as e:
            logger.error(f"Error writing notebook (generation {generation}): {e}")


def _combine_documents(
    docs, document_prompt=PromptTemplate.from_template(template="{page_content}"), document_separator="\n\n"  # type: ignore
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)
