import logging
import openai
from textwrap import dedent
from typing import Optional, OrderedDict
import json
import random
from human import Human

prompt_all = dedent("""\
    Your task is turn the following biographical data into the vivid biography of an almost real person.

    {stats_block_valued}

    Also recall the relevant historical, geographic, socioeconomic and cultural context, and make up whatever details you need. Unleash your creativity!

    Put the fictional biography (1000 words) into a markdown code block:

    ```markdown
    John Doe was born in 1900 in New York City. He was the son of a wealthy banker and a housewife.
    As a sickly child...
    ```

    Next, fill in this key data, which will be used to index the biography. Again use a markdown code block:

    ```markdown
    Name: John Doe
    Exact Birth Date: 1900/01/01
    Exact Death Date: 2000/07/13
    Home Area: New York City
    ```

    Finally, write an image prompt (maximum 900 characters) for Dall-E 3, in order to create a photograph to accompany the biography.
    Put your answer into a final markdown code block:

    ```markdown
    Photograph of a 30 year old man standing in New York of 1930. The man is a banker and wears a suit. In the background the Empire State Building is under construction.
    ```

    Take a deep breath and think step by step.\
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

def html_llm_biography(human: Human, model = "gpt-4", max_tokens = 7000) -> dict:
    if human.vars_html.get("Biography") is not None:
        logging.info(f"Variable Biography already sampled.")
        return

    stats_block = "\n".join([
        f"- {var}: {val}" if val is not None else f"- {var}: ?"
        for var, val in human.vars_stat.items()
    ])
    prompt = prompt_all.format(
        stats_block_valued=stats_block
    )
    body = query_llm(prompt, model, max_tokens)

    # extract the markdown blocks
    import re
    blocks = re.findall(r"```markdown\n(.*?)\n```", body, re.DOTALL)
    assert len(blocks) == 3
    human.vars_html["Biography"] = blocks[0]
    human.vars_html["Title Data"] = blocks[1]
    human.vars_html["Image Prompt"] = blocks[2]
    human.save()

    # key variables
    keyval = re.findall(r"(.+?): (.+)", blocks[1])
    keyval = {k.lower(): v for k, v in keyval}
    human.vars_html["Title Data"] = keyval
    human.save()
