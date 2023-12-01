from pathlib import Path

import openai
import backoff 
import cohere
import time
import logging
import pandas as pd

from dotenv import dotenv_values

base_path = Path("/cluster/home/afelderer/Thesis")
config = dotenv_values("/cluster/home/afelderer/Thesis/config/.env") # take environment variables from .env.

openai.organization = config["OPEN_AI_ORG"]
openai.api_key = config["OPEN_AI_TOKEN"]
co = cohere.Client(config["COHERE_API_KEY"])

providers = ["cohere", "openai"]

labels = {
    "loss": "a financial/business loss, or adverse business development experienced by the reporting entity",
    "unexpected": "an unexpected event or development experienced by, or related to the reporting entity"
}

query_template = """Does the following sequence contain (True) or not (False) a statement which describes or refers to {label}?
    Sequence: {sequence}
    Label:"""

system_message = "You are a research assistant paid for labeling the following statements as accurately as possible. Please return the label as True or False."


annotations = pd.read_pickle(base_path/"data/labeling/Eval-LLMs.pkl")
if "labeled" not in annotations.columns:
    annotations["labeled"] = False

for l in labels.keys():
    for provider in providers:
        if f"{provider}_{l}" not in annotations.columns:
            annotations[f"{provider}_{l}"] = None

@backoff.on_exception(backoff.expo, openai.error.RateLimitError, on_backoff=lambda x: print(f"""Backing off: {round(x['wait'])} seconds"""))
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def main():
    logging.basicConfig(filename='llms.log', level=logging.INFO)
    for i, x in annotations[~annotations.labeled].iterrows():
        for l in labels.keys():
            for provider in providers:
                if provider == "openai":
                    response = completions_with_backoff(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": query_template.format(sequence=x.text, label=labels[l])},
                        ],
                        temperature=0,
                        max_tokens=10)
                    annotations.loc[i, f"{provider}_{l}"] = response.choices[0].message.content
                elif provider == "cohere":
                    response = co.generate( 
                        model="command",
                        prompt= system_message + query_template.format(sequence=x.text, label=labels[l]),
                        num_generations=1,
                        temperature=0,
                        max_tokens=10,
                        stop_sequences=["\n"])
                    annotations.loc[i, f"{provider}_{l}"] = response.generations[0].text.strip()
            time.sleep(21)
        annotations.loc[i, "labeled"] = True
        annotations.to_pickle(base_path/"data/labeling/Eval-LLMs.pkl")
        logging.info(f"Finished labeling {i}.")

if __name__ == "__main__":
    main()
