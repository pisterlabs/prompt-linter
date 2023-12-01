"""Find character attributes described in a random sample of passages

Input
-----
script-passages
    path = mica-character-attribute-extraction/script-passages.csv
    csv file containing imdb-id, passage-id, passage, and characters fields

book-passages
    path = mica-character-attribute-extraction/book-passages.csv
    csv file containing book, passage-id, passage, and characters fields

Output
-----
attributes
    path = mica-character-attribute-extraction/passage-attributes.csv
    csv file containing story-id, passage-id, passage, characters, attributes
    story-id could be imdb-id or book name
    attributes are (type, value) pairs

Parameters
-----
model
    type of gpt model to use

sample
    sample size for each source

seed
    random seed

calculate-cost
    boolean flag that you can set to estimate the cost
"""

from lib import openai_prompting

import os
import re
import pandas as pd
import tqdm
from copy import deepcopy
from absl import flags
from absl import app

# define command-line flags
FLAGS = flags.FLAGS
flags.DEFINE_string("model", default="gpt-4-turbo-1106-preview", help="OpenAI model to use for prompting")
flags.DEFINE_integer("sample", default=1000, help="Number of story segments to sample")
flags.DEFINE_integer("seed", default=99, help="Seed for random sampling")
flags.DEFINE_bool("cost", default=False, help="Estimate cost only")
flags.DEFINE_float("input_rate", default=3e-6, help="model token rate for input tokens")
flags.DEFINE_float("output_rate", default=6e-6, help="model token rate for input tokens")

# directories and files
data_dir = os.path.join(os.getenv("DATA_DIR"), "mica-character-attribute-extraction")
scripts_file = os.path.join(data_dir, "script-passages.csv")
books_file = os.path.join(data_dir, "book-passages.csv")
attributes_file = os.path.join(data_dir, "passage-attributes.csv")

def prompt_character_attributes(_):

    # read passages
    scripts_df = pd.read_csv(scripts_file, index_col=None)
    books_df = pd.read_csv(books_file, index_col=None)

    # sample story segments
    sampled_scripts_df = scripts_df.sample(FLAGS.sample, random_state=FLAGS.seed)
    sampled_books_df = books_df.sample(FLAGS.sample, random_state=FLAGS.seed)
    sampled_scripts_df.rename(columns={"imdb-id": "story-id"}, inplace=True)
    sampled_books_df.rename(columns={"book": "story-id"}, inplace=True)
    passages_df = pd.concat([sampled_scripts_df, sampled_books_df])
    print(f"{len(passages_df)} passages will be prompted")

    # create the message template
    messages_template = [
        {
            "role": "user",
            "content": ("Find the character attributes in the following passage and write them as (character, "
                        "attribute-type, attribute-value) tuples in a numbered list. The attribute-type text should "
                        "be as brief and concise as possible.\nPassage: ")
        }
    ]

    # run the prompts or estimate the cost
    estimated_cost = 0
    actual_cost = 0
    attributes = []
    completions = []

    for segment in tqdm.tqdm(passages_df["passage"], unit="passage", desc="prompting"):
        messages = deepcopy(messages_template)
        segment = re.sub("\s+", " ", segment.strip())
        messages[0]["content"] += segment
        if FLAGS.cost:
            estimated_cost += openai_prompting.estimate_cost(messages, FLAGS.model, FLAGS.input_rate,
                                                             FLAGS.output_rate, 50)
        else:
            n_prompt_tokens = openai_prompting.num_tokens_from_messages(messages, FLAGS.model)
            completion = openai_prompting.prompt_sample(messages, FLAGS.model, max_tokens=n_prompt_tokens + 256)
            if completion is not None:
                attributes.append(completion.choices[0].message.content)
                completions.append(completion.model_dump_json())
                actual_cost += (completion.usage.completion_tokens * FLAGS.input_rate
                                + completion.usage.prompt_tokens * FLAGS.output_rate)
            else:
                attributes.append("")
                completions.append("")

    # print the cost, save the output
    if FLAGS.cost:
        print(f"Estimated cost = ${estimated_cost}")
    else:
        print(f"Cost incurred = ${actual_cost}")
        passages_df["attributes"] = attributes
        passages_df["completions"] = completions
        passages_df.to_csv(attributes_file, index=False)
    
if __name__ == '__main__':
    app.run(prompt_character_attributes)