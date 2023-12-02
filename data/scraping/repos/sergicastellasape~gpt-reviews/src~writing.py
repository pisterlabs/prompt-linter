import logging
import openai
import os
import re
import time

from config import ARGS

openai.api_key = os.getenv('OPENAIKEY')

def write(system_prompt,
          user_prompt_template,
          substitutions,
          script_path,
          temperature,
          parsing_options):
    # this function will always check if there's a script cached in the
    # script_path, and only generate if there isn't one!
    if not os.path.exists(script_path):
        # prompts have variables that are replaced here
        for k, v in substitutions.items():
            user_prompt_template = user_prompt_template.replace(k, v)
        prompt = user_prompt_template
        logging.info(f"Generating script for {script_path}")
        completion = generate(system_prompt, prompt, ARGS.model, temperature=temperature)
        script = parse_gpt_output(completion, parsing_options)
        assert script != ""
        with open(script_path, "w") as f:
            f.write(script)
    else:
        logging.info(f"Loading script from {script_path}")
        with open(script_path) as f:
            script = f.read()
    return script


def generate(system_prompt, user_prompt, model, temperature=0.7):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    logging.debug(f"Calling OpenAI with {messages}")
    retries = 3  # sometimes openAI just fails
    for i in range(retries):
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
        except Exception as e:
            if i == (retries - 1):
                raise
            else:
                logging.info(f"Retrying after exception {e}")
                time.sleep(3)
                continue
    return completion.choices[0].message['content']


def parse_gpt_output(text, parsing_options):
    if parsing_options.get("delimiter"):
        text = text.split(parsing_options["delimiter"])[-1].strip()
    if parsing_options.get("delete_gio"):
        text = text.replace("Giovani", "").replace("Gio", "")
    if parsing_options.get("delete_parenthesis"): # and brakets and asterisks
        text = re.sub(r"[\(\[\*].*?[\)\]\*]", "", text)
    return text
