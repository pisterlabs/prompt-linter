import logging
import openai
from textwrap import dedent
from typing import Optional, OrderedDict
import csv
import random
from human import Human

#model = "gpt-3.5-turbo"
#model = "gpt-3.5-turbo-instruct"
model = "gpt-4"
max_tokens = 3000

prompt_template = dedent("""\
    For a research project I have investigated the subset of all humans ever who fit the following criteria:

    {stats_block_valued}

    Please estimate the conditional distribution of another variable:

    - {var_name}

    Provide likely values, as well as an estimate of their relative frequency in the selected subset of all humans ever born.
    Provide your answer as a csv table with two columns (values and odds), for example:

    ```csv
    "left handed", 1.0
    "right handed", 8.5
    ```

    This csv table is parsed automatically by a python script, so please make sure it is formatted correctly.

    Other assistants report that discussing the historical context of the selected subpopulation has helped them make more accurate guesses.
    They also did best when thinking in population numbers, and ignoring irrelevant aspects like historical significance or moral judgements, instead focusing on hard data.
    In general, use your common sense, factual knowledge, and reasoning ability to come up with your best estimate for the conditional distribution.
    
    Think step by step.\
""")

def query_llm(prompt: str, model: str, max_tokens: int):
    if model in ["gpt-4", "gpt-3.5-turbo"]:
        kwarg = { 
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        }
        logging.info(f"Sending to LLM")
        logging.info(f"{kwarg}")
        response = openai.ChatCompletion.create(
            **kwarg
        )
        logging.info(f"Received from LLM")
        logging.info(f"{response}")
        body = response["choices"][0]["message"]["content"]
    elif model in ["gpt-3.5-turbo-instruct"]:
        kwarg = {
            "model": model,
            "max_tokens": max_tokens,
            "prompt": prompt
        }
        logging.info(f"Sending to LLM")
        logging.info(f"{kwarg}")
        response = openai.Completion.create(
            **kwarg
        )
        logging.info(f"Received from LLM")
        logging.info(f"{response}")
        body = response["choices"][0]["text"]
    else:
        raise NotImplementedError

    # cost estimate
    costs = {"gpt-4": 0.03, "gpt-3.5-turbo": 0.0015, "gpt-3.5-turbo-instruct": 0.0015}
    cost_estimate = costs[model] * response["usage"]["total_tokens"] / 1000
    logging.info(f"Cost Estimate: ${cost_estimate:.4f}")

    return body

def sample_llm(human: Human, var_name: str, model = "gpt-4", max_tokens = 3000):
    """
    Sample a variable using LLM.
    """
    if human.vars_stat.get(var_name) is not None:
        logging.info(f"Variable {var_name} already sampled.")
        return

    # format
    stats_block_valued = "\n".join([
        f"- {var}: {val}" if val is not None else f"- {var}" 
        for var, val in human.vars_stat.items()
        if val is not None
    ])
    prompt = prompt_template.format(
        stats_block_valued=stats_block_valued, 
        var_name=var_name
    )

    # query LLM
    body = query_llm(prompt, model, max_tokens)
    
    # parse response
    vals = []
    odds = []
    
    table = body.split("```csv\n")[-1].split("\n```")[0]
    logging.info(f"Table: {table}")
    for row in table.split("\n"):
        if row.startswith("#") or row.strip() == "":
            continue # skip comments and empty lines

        # use csv tokenizer to handle quotes and commas
        row = next(csv.reader([row], delimiter=',', quotechar='"'))
        if len(row) != 2:
            raise Exception(f"Could not parse row: {row}")
        val, odd = row
        val = val.strip()
        odd = odd.strip()
        try:
            odd = float(odd)
        except Exception as e:
            logging.error(f"Could not parse row: {row}")
            raise e

        vals.append(val)
        odds.append(odd)

    logging.info(f"Sampling distribution: {vals} {odds}")
    val = random.choices(vals, weights=odds)[0]
    logging.info(f"Sampled value: {var_name}={val}")

    # store value
    human.vars_stat[var_name] = val
    human.save()