import openai
import numpy as np
import random


def authenticate(args):
    with open(args.api_key) as f:
        api_key = f.readlines()[0].strip()
    openai.api_key = api_key
    return api_key


def seed_function(args):
    random.seed(args.seed)
    np.random.seed(args.seed)


def get_prompt(
    args,
    train_dataset,
    instruction_template,
    demonstration_template,
    demonstration_sep="\n",
):
    if not args.few_shot:
        prompt = instruction_template
    else:
        # Include examples in the prompt
        assert (
            train_dataset is not None
        ), "Want to do few-shot, but no train dataset provided"
        # Sample some examples from the train dataset
        collated_demonstrations = ""
        cols = train_dataset.column_names
        for example_id in range(len(train_dataset[cols[0]])):
            example = {col: train_dataset[col][example_id] for col in cols}
            cur_demonstration = demonstration_template.format(**example)
            collated_demonstrations = (
                collated_demonstrations + demonstration_sep + cur_demonstration
            )
        prompt = "{}\n{}".format(instruction_template, collated_demonstrations)
    return prompt, collated_demonstrations
