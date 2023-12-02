"""Process  census metadata with OpenAI
"""

import json
from random import shuffle

__author__ = "eric@civicknowledge.com"


def make_prompt_1(d):
    jd = json.dumps(d, indent=4)

    prompt = f"""
For each dict entry in the following JSON,  replace the <question> with  a question  that could be answered with a
dataset that is described by the "description" and return the new JSON. The question should include the text in
the "time" and "place". Questions should start start with a variety of phrases such as 'how many', 'which state has',
'in which year', 'what is', 'comparing', 'make a list of', 'show a map of' and 'where'.

{jd}
""".strip()

    return prompt


def make_prompt(d):
    prompt = f"""
Write a question that can be answered by analyzing data that is described by the following measurement description.
Questions should start start with a variety of phrases such as 'how many', 'in which year',
'what is', 'comparing', 'make a list of', 'show a map of' and 'where'. Try to not start the question with "which"

{d['measure']} for {d['restriction']} {d['time']} {d['place']}
""".strip()

    return prompt


def write_question(d):
    """Call the OpenAI completions interface to re-write the extra path for a census variable
    into an English statement that can be used to describe the variable."""

    import os

    import openai

    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=make_prompt(d),
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0.2,
        presence_penalty=0.2,
    )

    return response


def questions_tasks(mdf):
    """Write data records for generating questions from columns"""

    from researchrobot.datadecomp.census_demo_conditions import (
        random_place,
        random_time,
    )

    vds = []

    for idx, r in mdf.sample(frac=1).iterrows():
        d = {
            "column_id": str(r.uid),
            "measure": r.col_desc,
            "restriction": r.rest_description,
            "description": "<description>",
            "time": random_time(),
            "place": random_place(),
            "question": "<question>",
        }

        vds.append(d)

    shuffle(vds)
    return vds


def store_question_responses(task, response, db):
    r = response["choices"][0]["text"].strip()

    task["question"] = r

    db[task["column_id"]] = task
