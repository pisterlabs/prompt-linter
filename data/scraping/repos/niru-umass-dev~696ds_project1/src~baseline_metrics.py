"""trivial_baseline_utils.py: Contains utility methods for DS, BertScore, and Paraphrase analysis. __main__ also creates data/combined_data.json"""
import json
import os
from os import listdir
from tqdm.auto import tqdm
import openai
import time
import sys
import rouge
from collections import Counter
import matplotlib.pyplot as plt
from statistics import stdev
from scipy.stats import spearmanr
import numpy as np
from numpy import mean
import bert_score
from typing import List, Dict


def get_evaluator():
    return rouge.Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False, apply_avg=True, stemming=True,
                       ensure_compatibility=True)


def get_scorer():
    return bert_score.BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en", rescale_with_baseline=True)


def stem(x, evaluator):
    return Counter(evaluator.stem_tokens(evaluator.tokenize_text(x.lower())))


def calc_ds(summ_a, summ_b, summ_comm, evaluator):
    s_a, s_b, s_c = stem(summ_a, evaluator), stem(summ_b, evaluator), stem(summ_comm, evaluator)
    nr = sum((s_a & s_b).values()) + sum((s_a & s_c).values()) + sum((s_b & s_c).values()) - 2.0 * sum(
        (s_a & s_b & s_c).values())
    dr = sum((s_a | s_b | s_c).values())
    return 1.0 - (nr / dr)


def calc_ds_pair(summ_a, summ_b, evaluator):
    s_a, s_b = stem(summ_a, evaluator), stem(summ_b, evaluator)
    nr = sum((s_a & s_b).values())
    dr = sum((s_a | s_b).values())
    return 1.0 - (nr / dr)


def calc_bs_pair(seq_a, seq_b, scorer):
    ab = [s.detach().numpy()[0] for s in scorer.score([seq_a], [seq_b])]
    # ba = [s.detach().numpy()[0] for s in scorer.score([seq_a], [seq_b])]
    # a_b = (np.array(ab) + np.array(ba)) / 2.0
    a_b = np.array(ab)
    return a_b


def calc_bs_triplet(summ_a, summ_b, summ_comm, scorer):
    ab = [s.detach().numpy()[0] for s in scorer.score([summ_a], [summ_b])]
    ba = [s.detach().numpy()[0] for s in scorer.score([summ_a], [summ_b])]
    a_comm = [s.detach().numpy()[0] for s in scorer.score([summ_a], [summ_comm])]
    comm_a = [s.detach().numpy()[0] for s in scorer.score([summ_comm], [summ_a])]
    b_comm = [s.detach().numpy()[0] for s in scorer.score([summ_b], [summ_comm])]
    comm_b = [s.detach().numpy()[0] for s in scorer.score([summ_comm], [summ_b])]
    a_b = (np.array(ab) + np.array(ba)) / 2.0
    a_c = (np.array(a_comm) + np.array(comm_a)) / 2.0
    b_c = (np.array(b_comm) + np.array(comm_a)) / 2.0
    return (a_b + a_c + b_c) / 3.0


def compare_orig_para(x, y, orig_type, score_type='DS'):
    print()
    print(f"{score_type} | {orig_type} vs paraphrase".upper())
    # x = ds_gen_orig
    # y = ds_gen_para
    x = [100 * value for value in x]
    y = [100 * value for value in y]
    plt.scatter(x, y)
    plt.show()

    r = np.corrcoef(x, y)

    rankcorr = spearmanr(x, y)

    diffs = [x[i] - y[i] for i in range(len(x))]

    print('\n', "Separate Statistics", '\n')
    print(f"{orig_type}-orig | µ({score_type}) = {np.mean(x):.1f} | σ({score_type}) = {np.std(x):.1f}")
    print(f"{orig_type}-para | µ({score_type}) = {np.mean(y):.1f} | σ({score_type}) = {np.std(y):.1f}")
    print()
    print("Combined Statistics", '\n')
    print(f"Spearman Rank Correlation = {rankcorr}")
    print(f"Correlation = {r[0, 1]:.2f}")
    print(f"µ({orig_type}-para) = {np.mean(diffs):.1f} | σ({orig_type}-para) = {np.std(diffs):.1f}", '\n')


def get_ds_scores(combined_data: List[Dict], summ_type, evaluator, compute, negation=False):
    records = []
    for example_id, example in enumerate(combined_data):
        split = example['split']
        if summ_type == 'gen' and split == 'train':
            continue

        if summ_type == 'gen':
            cont_a, cont_b, cont_comm = example['gen_a'], example['gen_b'], example['gen_comm']
        if summ_type == 'ref':
            cont_a, cont_b, cont_comm = " ".join(example['refs_a'][0]), " ".join(example['refs_b'][0]), " ".join(example['refs_comm'][0])
        if negation == True:
            cont_a, cont_b = " ".join(example['refs_a'][0]), " ".join(example['refs_b'][0])

        if compute == "pair":
            example_ds_score = calc_ds_pair(cont_a, cont_b, evaluator)
            records.append((summ_type, split, example_id, cont_a, cont_b, example_ds_score))

        else:
            example_ds_score = calc_ds(cont_a, cont_b, cont_comm, evaluator)
            records.append((summ_type, split, example_id, cont_a, cont_b, cont_comm, example_ds_score))
        # records.append((summ_type, split, example_id, cont_a, cont_b, cont_comm, example_ds_score))
    return records


def get_bs_scores(combined_data: List[Dict], summ_type, scorer, compute, negation=False):
    records = []
    for example_id, example in enumerate(combined_data):
        split = example['split']
        if summ_type == 'gen' and split == 'train':
            continue

        if summ_type == 'gen':
            cont_a, cont_b, cont_comm = example['gen_a'], example['gen_b'], example['gen_comm']
        if summ_type == 'ref':
            cont_a, cont_b, cont_comm = " ".join(example['refs_a'][0]), " ".join(example['refs_b'][0]), " ".join(example['refs_comm'][0])
        if negation == True:
            cont_a, cont_b = " ".join(example['refs_a'][0]), " ".join(example['refs_b'][0])

        if compute == "pair":
            example_bs_score = calc_bs_pair(cont_a, cont_b, scorer)
            records.append((summ_type, split, example_id, cont_a, cont_b, example_bs_score[2]))
        else:
            example_bs_score = calc_bs_triplet(cont_a, cont_b, cont_comm, scorer)
            records.append((summ_type, split, example_id, cont_a, cont_b, cont_comm, example_bs_score[2]))
        # records.append((summ_type, split, example_id, cont_a, cont_b, cont_comm, example_bs_score[2]))
    return records


def get_baseline_scores(dataset_path: str, is_paraphrase: bool, orig_type: str, score_fn_name: str = "calc_ds"):
    score_fn = globals()[score_fn_name]
    assert orig_type in {'gen', 'refs'}
    dataset_type = 'para_' if is_paraphrase else ''

    dataset = json.load(open(dataset_path, 'r'))

    ds_scores = []
    # The range is conditional on what splits are present
    for example_no in (range(20, 48) if len(dataset) == 48 else range(28)):
        orig_example = dataset[example_no]
        # print(f"Example {example_no}")
        summ_a = orig_example[f'{dataset_type}{orig_type}_a']
        summ_b = orig_example[f'{dataset_type}{orig_type}_b']
        summ_comm = orig_example[f'{dataset_type}{orig_type}_comm']

        if orig_type == 'gen':
            ds_scores.append(score_fn(summ_a, summ_b, summ_comm))
        elif orig_type == 'refs':
            ds_scores.append(mean(
                [
                    score_fn(summ_a[idx], summ_b[idx], summ_comm[idx]) for idx in range(len(summ_a))
                ]
            )
            )

    return ds_scores


if __name__ == '__main__':
    pass
    # print(f"main_path = {sys.argv[0]}")
    # project_folder = os.getcwd()[:os.getcwd().find('696ds_project1')+ len('696ds_project1')]
    # data_folder = os.path.join(project_folder, 'data')
