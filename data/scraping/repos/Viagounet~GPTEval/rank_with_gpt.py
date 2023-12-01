import ast
import time
import pandas as pd
import random
import openai
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

def rank(summaries):
    """Mock ranking function. Returns a random ranking for each summary."""
    print(summaries)
    print("\n\n\n")
    return [random.randint(1, len(summaries)) for _ in summaries]

# Load the dataset
ordered_letters = "ABCDEFGHIJKLMNOP"
data = pd.read_csv('summeval_dataset.csv', delimiter='\t', error_bad_lines=False)
stored_rankings = []
stored_tokens = []
for machine_summaries, human_summaries in zip(data["machine"].tolist(),
                                              data["human"].tolist()):
    machine_summaries = ast.literal_eval(machine_summaries)
    human_summaries = ast.literal_eval(human_summaries)
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    random.shuffle(letters)
    named_summaries = {}
    for i, machine_summary in enumerate(machine_summaries):
        named_summaries[letters[i]] = (machine_summary, i)
    machines_string = "\n\n".join(f"Machine Summary {letter} : {named_summaries[letter][0]}" for letter in ordered_letters)
    humans_string = "\n\n".join(f"Human Summary {i} : {human_summaries[i]}" for i in range(len(human_summaries)))

    gpt_prompt = f"""<Human summaries>
{humans_string}
</Human summaries>
<Machine summaries>
{machines_string}
</Machine summaries>
-------
Rank the machine summaries according to how relevant they are to the human summaries.
Rank them by referencing their letters. Example : Ranking=X>Z>W>[...]>R.
Always use ">", do not use any other symbol to compare them.
------
Ranking=
"""
    completion = openai.ChatCompletion.create(
    model="gpt-4",
    temperature=0,
    messages=[
        {"role": "system", "content": gpt_prompt},
    ]
    )
    rankings = completion["choices"][0]["message"]["content"]
    rankings = [named_summaries[letter][1] for letter in rankings.split(">")]
    tokens = completion["usage"]["total_tokens"]
    print(len(stored_rankings)+1)
    stored_rankings.append(rankings)
    stored_tokens.append(tokens)
    time.sleep(3)
data["rankings"] = stored_rankings
data["tokens"] = stored_tokens
data.to_csv("rankings.csv", sep="\t")