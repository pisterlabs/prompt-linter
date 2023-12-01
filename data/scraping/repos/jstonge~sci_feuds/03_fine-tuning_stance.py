import re
from pathlib import Path

import numpy as np
import openai
import pandas as pd
import tiktoken

from helpers import map2int_3, map2int_5, map2int_9

ROOT_DIR = Path("..")
OUTPUT_DIR = ROOT_DIR / 'output'
GROBID_DIR = OUTPUT_DIR / 'group_selection_grobid'
SPACY_DIR = OUTPUT_DIR / 'spacy_group_selection_grobid'
STANCE_DIR = OUTPUT_DIR / 'stance_detection'
CCA_DIR = OUTPUT_DIR / 'cca'

openai.api_key = open("myapikey.txt", "r").read().strip("\n")
enc = tiktoken.encoding_for_model("curie")

def max_wc(df, thresh=550):
    return df[df.prompt.str.split(" ").map(len) < thresh]


def prep_data(n_classes=3, balanced=True, thresh_wc=550):
    df = pd.read_json("../output/stance_detection/all.json")
    
    assert n_classes in [3,5,9], "only 3,5,9 implemented"
    
    if n_classes == 3:
        # when only 3 clases, we only keep the most "representative" paragraph
        # based on stance estimated by Beese's model .
        df = df[((df.stance > 0.75) | (df.stance < -0.25)) | ((df.stance > -.25) & (df.stance < .25) ) ]
        df['stance_discrete'] = df.stance.map(map2int_3)
    elif n_classes == 5:
        df['stance_discrete'] = df.stance.map(map2int_5)
    else:
        df['stance_discrete'] = df.stance.map(map2int_9)            

    if balanced:
        min_cat = df.value_counts("stance_discrete").min()
        df = df.sample(frac=1).groupby("stance_discrete").head(min_cat)

    # prepare data for openai's fine tuning format
    prompt = df.abstract.map(lambda x: str(x) + "\n\n###\n\n")
    completion = df['stance_discrete'].map(lambda x: " " + x)
    df_prompt_completion = pd.DataFrame(zip(prompt, completion), columns=["prompt", "completion"])

    # threshold word count
    df_prompt_completion = max_wc(df_prompt_completion, thresh=thresh_wc)

    df_prompt_completion.value_counts("completion")

    # save to `output/stance_detection` directory
    df_prompt_completion.to_json(STANCE_DIR / "stance_detection_unbalanced_5catego_2023-05-08.jsonl", lines=True, orient='records')


# Prepare data using openAI cli
#! openai tools fine_tunes.prepare_data -f stance_detection_2023-05-08.jsonl
#! openai tools fine_tunes.prepare_data -f stance_detection_balanced_5catego_2023-05-08.jsonl

# Inspect prepared data to make sure all categories are in both files
# df_train = pd.read_json(STANCE_DIR / "stance_detection_2023-05-08_prepared_train.jsonl", lines=True, orient="records")
# df_valid = pd.read_json(STANCE_DIR / "stance_detection_2023-05-08_prepared_valid.jsonl", lines=True, orient="records")

# df_train.value_counts("completion")
# df_valid.value_counts("completion")


# Fine-tuning model using openai cli
# openai api fine_tunes.create -t "stance_detection_2023-05-08_prepared_train.jsonl" \                         
#                              -v "stance_detection_2023-05-08_prepared_valid.jsonl" \
#                              -m "curie:ft-personal-2023-05-01-19-28-15" \
#                              --compute_classification_metrics \
#                              --classification_n_classes 3








# Validating fine-tuned model
# !openai wandb sync

best_model_so_far_3 = "curie:ft-personal-2023-05-08-21-10-03"
# best_model_so_far_5 = "curie:ft-personal-2023-05-09-23-14-24"
best_model_so_far_5 = "curie:ft-personal-2023-05-08-19-17-01" 

best_model_so_far_id_3 = "ft-F2HCJCsDcXnUodzu37pongJf"   # 3/balanced/curie
# best_model_so_far_id_5 = "ft-fWXcntKe6ctUxxG2GANyneXJ" # 5/balanced/curie
best_model_so_far_id_5 = "ft-FrUEGnWr4iGrIPpuYKDunfJ0" # 5/unbalanced/curie*
best_model_so_far_id_5 = "ft-CaL8qTKKA3gtbSWsVmr1vDQU" # 5/unbalanced/ada

def simple_validate(n_classes=5):
    """only 5 classes model. When n_classes=3 and balanced, the random pick = 33%"""
    df_prompt_completion = pd.read_json(STANCE_DIR / "stance_detection_5catego_2023-05-08.jsonl", lines=True, orient='records')

    np.sum(1 == df_prompt_completion['completion']) / len(df_prompt_completion)
    np.sum(2 == df_prompt_completion['completion']) / len(df_prompt_completion)
    np.sum(3 == df_prompt_completion['completion']) / len(df_prompt_completion)
    np.sum(4 == df_prompt_completion['completion']) / len(df_prompt_completion)
    np.sum(5 == df_prompt_completion['completion']) / len(df_prompt_completion)


res = pd.read_csv(STANCE_DIR / f"classif_metric_{best_model_so_far_id_5}.csv")

res[res['classification/accuracy'].notnull()].tail(1) # acc(3cat)=75%; acc(5cat but unbalanced)=64% (same weighted_f1_score)
# F1-score: (2 * (Precision * Recall))/(Precision + Recall); is it the same than 
res[res['classification/weighted_f1_score'].notnull()]['classification/weighted_f1_score'].plot()

# About wandb:
# see https://wandb.ai/borisd13/GPT-3/reports/GPT-3-exploration-fine-tuning-tips--VmlldzoxNDYwODA2?utm_source=fully_connected&utm_medium=blog&utm_campaign=openai+gpt-3
# But in a nutshell:
#  - loss should go down. Not too much oscillations.
#  - We want a high validation token accuracy
#  - learing rate multiplier is an important hyperparameter.