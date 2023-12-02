import os
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from tqdm import tqdm
import pandas as pd
import sys
from IPython.display import display, HTML
from typing import List
from sasc.modules.emb_diff_module import EmbDiffModule
import numpy as np
import matplotlib
import imodelsx.util
from copy import deepcopy
import re
import sasc.generate_helper
import sasc.viz
import scipy.special
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from sasc.evaluate import D5_Validator
import openai
from sasc.modules.fmri_module import fMRIModule
from pprint import pprint
import joblib
from sasc.config import RESULTS_DIR
import torch.cuda
import scipy.special


def explanation_story_match(EXPT_DIR, expls, paragraphs, prompts):
    if os.path.exists(join(EXPT_DIR, "story_data_match.pdf")):
        return
    val = D5_Validator()

    # visualize single story
    scores_data_story = sasc.viz.get_story_scores(val, expls, paragraphs)
    joblib.dump(scores_data_story, join(EXPT_DIR, "scores_data_story.pkl"))
    s_data = sasc.generate_helper.viz_paragraphs(
        paragraphs,
        scores_data_story,
        expls,
        prompts,
        normalize_to_range=True,
        moving_average=True,
        shift_to_range=True,
    )
    with open(join(EXPT_DIR, "story.html"), "w") as f:
        f.write(s_data.encode("ascii", errors="ignore").decode())

    # compute scores heatmap
    # print('expls', expls, 'paragraphs', paragraphs)
    scores_mean, scores_all = sasc.generate_helper.compute_expl_data_match_heatmap(
        val, expls, paragraphs
    )
    joblib.dump(
        {"scores_mean": scores_mean, "scores_all": scores_all},
        join(EXPT_DIR, "scores_data.pkl"),
    )
    sasc.viz.heatmap(scores_mean.T, expls, ylab="Story", xlab="Explanation")
    # plt.savefig(join(EXPT_DIR, "story_data_match.png"), dpi=300)
    plt.savefig(join(EXPT_DIR, "story_data_match.pdf"), bbox_inches="tight")


def module_story_match(EXPT_DIR, expls, paragraphs, voxel_nums, subject):
    if os.path.exists(join(EXPT_DIR, f"scores_mod_ngram_length={0}.pkl")):
        return

    # compute with paragraphs overlapping into each other
    # sasc.generate_helper.compute_expl_module_match_heatmap
    (
        scores_mod,
        _,
        all_scores,
    ) = sasc.generate_helper.compute_expl_module_match_heatmap_cached_single_subject(
        expls, paragraphs, voxel_nums, subject
    )
    joblib.dump(
        {
            "scores_mean": scores_mod,
            "scores_all": all_scores,
        },
        join(EXPT_DIR, f"scores_mod_ngram_length={0}.pkl"),
    )

    # make plot
    s = scores_mod.T
    s = scipy.special.softmax(s, axis=0)
    sasc.viz.heatmap(s, expls, ylab="Story", xlab="Module")

    diag_diff = (
        np.mean(np.diag(s))
        - (
            np.mean(s[np.triu_indices_from(s, k=1)])
            + np.mean(s[np.tril_indices_from(s, k=-1)])
        )
        / 2
    ).round(5)
    plt.title(os.path.basename(EXPT_DIR) + " diag_diff=" + str(diag_diff))

    plt.savefig(join(EXPT_DIR, f"story_module_match.pdf"), bbox_inches="tight")

    # with overlaps
    # ngram_lengths = [10, 50, 100, 384]
    # for i, ngram_length in enumerate(ngram_lengths):
    #     print(i, '/', len(ngram_lengths), 'ngram length', ngram_length)
    #     scores_mod, scores_max_mod, all_scores, all_ngrams = \
    #         notebook_helper.compute_expl_module_match_heatmap_running(
    #             expls, paragraphs, voxel_nums, subjects,
    #             ngram_length=ngram_length, paragraph_start_offset_max=50,
    #         )
    #     joblib.dump({
    #         'scores_mean': scores_mod,
    #         'scores_all': all_scores,
    #     }, join(EXPT_DIR, f'scores_mod_ngram_length={ngram_length}.pkl'))


def sweep_default_and_polysemantic(subjects=["UTS01", "UTS03"], setting="default"):
    EXPT_PARENT_DIR = join(RESULTS_DIR, "stories", setting)
    EXPT_NAMES = sorted(os.listdir(EXPT_PARENT_DIR))

    # filter EXPT_NAMES that don't contain any of the subjects
    EXPT_NAMES = [
        expt_name
        for expt_name in EXPT_NAMES
        if any([subject.lower() in expt_name for subject in subjects])
    ]

    for EXPT_NAME in EXPT_NAMES:
        EXPT_DIR = join(EXPT_PARENT_DIR, EXPT_NAME)
        rows = joblib.load(join(EXPT_DIR, "rows.pkl"))
        expls = rows.expl.values

        # original version
        if "paragraph" in rows.columns:
            paragraphs = rows.paragraph.values
            prompts = rows.prompt.values
        # new version
        else:
            prompts_paragraphs = joblib.load(
                join(EXPT_DIR, "prompts_paragraphs.pkl"),
            )
            prompts = prompts_paragraphs["prompts"]
            paragraphs = prompts_paragraphs["paragraphs"]

        voxel_nums = rows.module_num.values
        subjects = rows.subject.values

        # run things
        print("Computing expl<>story match", EXPT_NAME)
        explanation_story_match(EXPT_DIR, expls, paragraphs, prompts)
        torch.cuda.empty_cache()

        print("Computing module<>story match", EXPT_NAME)
        module_story_match(EXPT_DIR, expls, paragraphs, voxel_nums, subjects[0])
        torch.cuda.empty_cache()


def sweep_interactions(subjects=["UTS01", "UTS03"]):
    # iterate over seeds
    seeds = range(1, 8)
    setting = "interactions"
    for subject in subjects:  # ["UTS01", "UTS03"]:
        for seed in seeds:
            STORIES_DIR = join(RESULTS_DIR, "stories")
            EXPT_NAME = f"{subject.lower()}___jun14___seed={seed}"
            EXPT_DIR = join(STORIES_DIR, setting, EXPT_NAME)
            # rows = joblib.load(join(EXPT_DIR, "rows1.pkl"))
            # expls = rows.expl.values
            prompts_paragraphs = joblib.load(
                join(EXPT_DIR, "prompts_paragraphs.pkl"),
            )
            rows1_rep = joblib.load(join(EXPT_DIR, "rows.pkl"))
            prompts = prompts_paragraphs["prompts"]
            paragraphs = prompts_paragraphs["paragraphs"]
            voxel_nums = rows1_rep.module_num.values
            subjects = rows1_rep.subject.values
            expls = rows1_rep.expl.values
            print(
                f"Loaded {len(prompts)} prompts, {len(paragraphs)} paragraphs, {len(rows1_rep)} (repeated1) rows"
            )

            # run things
            print("Computing expl<>story match", EXPT_NAME)
            explanation_story_match(EXPT_DIR, expls, paragraphs, prompts)
            torch.cuda.empty_cache()

            print("Computing module<>story match", EXPT_NAME)
            module_story_match(EXPT_DIR, expls, paragraphs, voxel_nums, subjects[0])
            torch.cuda.empty_cache()


if __name__ == "__main__":
    sweep_interactions(subjects=["UTS01", "UTS03"])
    # sweep_default_and_polysemantic(subjects=['UTS01', 'UTS03'], setting="polysemantic")
    # sweep_default_and_polysemantic(subjects=['UTS01', 'UTS03'], setting="default")
