#!python3

import json
import os
import re
import signal
import sys
from datetime import datetime
from enum import Enum
from typing import Annotated, List
from rich.console import Console
from rich.markdown import Markdown


import subprocess
import typer
from icecream import ic
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from pydantic import BaseModel
from rich.console import Console

from openai_wrapper import choose_model, setup_gpt

console = Console()


# By default, when you hit C-C in a pipe, the pipe is stopped
# with this, pipe continues
def keep_pipe_alive_on_control_c(signum, frame):
    del signum, frame  # unused variables
    sys.stdout.write(
        "\nInterrupted with Control+C, but I'm still writing to stdout...\n"
    )
    sys.exit(0)


# Register the signal handler for SIGINT
signal.signal(signal.SIGINT, keep_pipe_alive_on_control_c)

original_print = print
is_from_console = False


gpt_model = setup_gpt()
app = typer.Typer()


# Todo consider converting to a class
class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# Shared command line arguments
# https://jacobian.org/til/common-arguments-with-typer/
@app.callback()
def load_options(
    ctx: typer.Context,
    u4: Annotated[bool, typer.Option] = typer.Option(False),
):
    ctx.obj = SimpleNamespace(u4=u4)


def process_shared_app_options(ctx: typer.Context):
    return ctx


# GPT performs poorly with trailing spaces (wow this function was writting by gpt)
def remove_trailing_spaces(str):
    return re.sub(r"\s+$", "", str)


@app.command()
def group(
    ctx: typer.Context,
    markdown: Annotated[bool, typer.Option()] = True,
):
    process_shared_app_options(ctx)
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))

    system_prompt = f"""You help group similar items into categories.  Exclude any linnes that are markdown headers. Output the category headers as markdown, and list the line items as list eelemnts below. Eg.

# Grouping A
* line 1
* line 2

IF possible, categories should match the following

- [Dealer of smiles and wonder](#dealer-of-smiles-and-wonder)
- [Mostly car free spirit](#mostly-car-free-spirit)
- [Disciple of the 7 habits of highly effective people](#disciple-of-the-7-habits-of-highly-effective-people)
- [Fit fellow](#fit-fellow)
- [Emotionally healthy human](#emotionally-healthy-human)
- [Husband to Tori - his life long partner](#husband-to-tori---his-life-long-partner)
- [Technologist](#technologist)
- [Professional](#professional)
- [Family man](#family-man)
- [Father to Amelia - an incredible girl](#father-to-amelia---an-incredible-girl)
- [Father to Zach - a wonderful boy](#father-to-zach---a-wonderful-boy)

     """
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(user_text),
        ],
    )
    model_name = "gpt-4-1106-preview"
    model = ChatOpenAI(model=model_name)

    ic(model_name)
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({})
    if markdown:
        console = Console()
        md = Markdown(response)
        console.print(md)
    else:
        print(response)


def patient_facts():
    return """
* Kiro is a co-worker
* Zach, born in 2010 is son
* Amelia, born in 2014 is daughter
* Tori is wife
* Physical Habits is the same as physical health and exercisies
* Bubbles are a joy activity
* Turkish Getups (TGU) is about physical habits
* Swings refers to Kettle Bell Swings
* Treadmills are about physical health
* 750words is journalling
* I work as an engineering manager (EM) in a tech company
* A refresher is a synonym for going to the gym
"""


def openai_func(cls):
    return {"name": cls.__name__, "parameters": cls.model_json_schema()}


@app.command()
def journal_report(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    responses: int = typer.Option(1),
    debug: bool = False,
    u4: Annotated[bool, typer.Option()] = True,
    journal_for: str = typer.Argument(
        datetime.now().date(), help="Pass a date or int for days ago"
    ),
):
    process_shared_app_options(ctx)

    # Get my closest journal for the day:
    completed_process = subprocess.run(
        f"python3 ~/gits/nlp/igor_journal.py body {journal_for} --close",
        shell=True,
        check=True,
        text=True,
        capture_output=True,
    )
    user_text = completed_process.stdout

    # remove_trailing_spaces("".join(sys.stdin.readlines()))

    # Interesting we can specify in the prompt or in the "models" via text or type annotations
    class Person(BaseModel):
        Name: str
        Relationship: str
        Sentiment: str
        SummarizeInteraction: str

    class Category(str, Enum):
        Husband = ("husband",)
        Father = ("father",)
        Entertainer = ("entertainer",)
        PhysicalHealth = "physical_health"
        MentalHealth = "mental_health"
        Sleep = "sleep"
        Bicycle = "bicycle"
        Balloon = "balloon_artist"
        BeingAManager = "being_a_manager"
        BeingATechnologist = "being_a_technologist"

    class CategorySummary(BaseModel):
        TheCategory: Category
        Observations: List[str]

    class Recommendation(BaseModel):
        ThingToDoDifferently: str
        ReframeToTellYourself: str
        PromptToUseDuringReflection: str
        ReasonIncluded: str

    class AssessmentWithReason(BaseModel):
        scale_1_to_10: int  # Todo see if can move scale to type annotation (condint
        reasoning_for_assessment: str

    class GetPychiatristReport(BaseModel):
        Date: datetime
        DoctorName: str
        PointFormSummaryOfEntry: List[str]
        Depression: AssessmentWithReason
        Anxiety: AssessmentWithReason
        Mania: AssessmentWithReason
        PromptsForCognativeReframes: List[str]
        PeopleInEntry: List[Person]
        Recommendations: List[Recommendation]
        CategorySummaries: List[CategorySummary]

    report = openai_func(GetPychiatristReport)

    system_prompt = f""" You are an expert psychologist named Dr {{model}} who writes reports after reading patient's journal entries

You task it to write a report based on the journal entry that is going to be passed in

# Here are some facts to help you assess
{patient_facts()}

# Report

* Include 2-5 recommendations
* Don't include Category Summaries for Categories where you have no data
"""

    process_shared_app_options(ctx)
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(user_text),
        ],
    )
    model_name = "gpt-4-1106-preview" if u4 else "gpt-3.5-turbo-1106"
    model = ChatOpenAI(model=model_name)
    ic(model_name)
    chain = (
        prompt
        | model.bind(function_call={"name": report["name"]}, functions=[report])
        | JsonOutputFunctionsParser()
    )

    response = chain.invoke({"model": model_name})
    with open(os.path.expanduser("~/tmp/journal_report/latest.json"), "w") as f:
        json.dump(response, f, indent=2)

    perma_path = os.path.expanduser(
        f"~/tmp/journal_report/{response['Date']}_{response['DoctorName']}.json".replace(
            " ", "_"
        )
    )
    with open(perma_path, "w") as f:
        json.dump(response, f, indent=2)
    print(json.dumps(response, indent=2))
    print(perma_path)


if __name__ == "__main__":
    app()
