import argparse
import csv
import os
import sys
import time
import traceback
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from coherence_manager import (ContextualizedCoherenceManager,
                               DummyCoherenceManager)
from efm import EFMReader
from explanation_generator import GreedySentenceSelector, ILPSentenceSelector
from opinion_contextualization import OpinionContextualizer, contextualize
from random_seed import set_random_seed
from sentence_pair_model import TfIdfSentencePair
from util import array2string


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        default="data/toy/train.csv",
        help="Input corpus path",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="data/toy/test.csv",
        help="Custom target file",
    )
    parser.add_argument(
        "-c",
        "--candidates",
        type=str,
        default=None,
        help="Custom file contains review IDs for sentence selection",
    )
    parser.add_argument(
        "-o", "--out", type=str, default="selected.csv", help="Ouput file path"
    )
    parser.add_argument(
        "-s",
        "--strategy",
        choices=[
            "greedy-efm",
            "ilp-efm",
        ],
        default="greedy-efm",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.5,
        help="Trace off factor between open review cost and representative sentence cost",
    )
    parser.add_argument(
        "-p",
        "--preference_dir",
        type=str,
        default="data/toy/efm",
        help="Preference path",
    )
    parser.add_argument(
        "-m", "--contextualizer_path", type=str, default="data/toy/asc2v/model.params"
    )
    parser.add_argument(
        "-rs", "--random_seed", type=int, default=None, help="Random seed value"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_path", type=str, default="dist")
    parser.add_argument("--debug_size", type=int, default=100)

    args = parser.parse_args()
    return args


SENTENCE_SELECTION_OUTPUT_HEADER = [
    "reviewerID",
    "asin",
    "id",
    "selected sentences",
    "sentences",
    "aspects",
    "demand",
    "selected reviews",
    "n_review",
    "n_sentence",
    "n_selected_reviews",
    "n_selected_sentences",
    "solve time",
    "total time",
    "objective value",
    "objective bound",
    "objective gap",
]


def get_candidates(corpus, user, item, demand, simplify=True):
    aspects = [aspect for aspect, count in demand.items() if count > 0]
    if simplify:
        candidates = corpus[(corpus["asin"] == item) & (corpus["aspect"].isin(aspects))]
    else:
        candidates = corpus
    simplified_demand = {}
    for aspect, count in demand.items():
        if count > 0:
            n_avail_sentences = len(candidates[candidates["aspect"] == aspect])
            if n_avail_sentences >= count:
                simplified_demand[aspect] = count
            elif n_avail_sentences > 0:
                simplified_demand[aspect] = n_avail_sentences
    return candidates, simplified_demand


def contextualize_candidate_sentences(candidates, user, contextualizer, top_k=10):
    candidates = candidates.copy()
    candidates["original reviewerID"] = candidates["reviewerID"]
    candidates["reviewerID"] = user
    candidates = contextualize(candidates, contextualizer, top_k=top_k)
    candidates["reviewerID"] = candidates["original reviewerID"]
    return candidates


def get_preference(preference_dir, preference_type="efm", verbose=False):
    from mter import MTERReader
    if 'mter' in preference_type:
        return MTERReader(preference_dir, verbose=verbose)
    return EFMReader(preference_dir, verbose=verbose)


def get_coherence_manager(preference=None, verbose=False):
    if preference is not None:
        return ContextualizedCoherenceManager(preference, verbose=verbose)
    else:
        return DummyCoherenceManager(verbose=verbose)


def get_contextualizer(contextualizer_path, preference, strategy, verbose=False):
    return OpinionContextualizer(
        contextualizer_path, preference, strategy=strategy, verbose=verbose
    )


def get_generator(
    generator_type,
    preference,
    sentence_pair_model,
    alpha=0.5,
    verbose=False,
):
    coherence_manager = get_coherence_manager(preference, verbose)
    if "greedy" in generator_type:
        return GreedySentenceSelector(
            coherence_manager, sentence_pair_model, alpha, generator_type, verbose
        )
    elif "ilp" in generator_type:
        return ILPSentenceSelector(
            coherence_manager, sentence_pair_model, alpha, verbose
        )


def get_corpus(input_path):
    corpus = pd.read_csv(input_path)
    corpus = corpus.reset_index()
    corpus["id"] = corpus["reviewerID"].map(str) + "-" + corpus["asin"].map(str)
    corpus["instance"] = corpus["id"].map(str) + "-" + corpus["sentence"].map(str)
    corpus.drop_duplicates("instance", inplace=True)
    corpus = corpus.drop(["instance"], axis=1)
    return corpus


def get_target_explanations(data_path, candidates_path=None):
    gt = pd.read_csv(data_path)
    gt["id"] = gt["reviewerID"].map(str) + "-" + gt["asin"].map(str)
    gt["instance"] = gt["id"].map(str) + "-" + gt["sentence"].map(str)
    gt.drop_duplicates("instance", inplace=True)
    target = (
        gt.groupby(["reviewerID", "asin", "id"])["reviewerID", "asin", "id"]
        .nunique()
        .drop(columns=["reviewerID", "asin", "id"])
        .reset_index()
    )
    if candidates_path and os.path.exists(candidates_path):
        df = pd.read_csv(candidates_path)
        candidates = df["id"].tolist()
        target = target[target["id"].isin(candidates)]
    target["aspects"] = target["id"].map(gt.groupby(["id"])["aspect"].apply(list))
    target["opinions"] = target["id"].map(gt.groupby(["id"])["opinion"].apply(list))
    target["opinion positions"] = target["id"].map(
        gt.groupby(["id"])["opinion_pos"].apply(list)
    )
    target["sentences"] = target["id"].map(gt.groupby(["id"])["sentence"].apply(list))
    return target


def generate_explanations(args):
    if args.random_seed:
        set_random_seed(args.random_seed)

    preference = get_preference(args.preference_dir, args.strategy, args.verbose)

    contextualizer = get_contextualizer(
        args.contextualizer_path,
        preference,
        strategy=args.strategy,
        verbose=args.verbose,
    )

    sentence_pair_model = TfIdfSentencePair(args.verbose)

    generator = get_generator(
        args.strategy,
        preference,
        sentence_pair_model,
        alpha=args.alpha,
        verbose=args.verbose,
    )

    corpus = get_corpus(args.input_path)

    target = get_target_explanations(args.target, args.candidates)

    if args.debug:
        print("Debug files are being saved at {}".format(args.debug_path))
        if not os.path.exists(args.debug_path):
            os.makedirs(args.debug_path)
        if args.debug_size > 0:
            target = target[: args.debug_size]

    with open(os.path.join(args.out), "w") as f:
        writer = csv.DictWriter(f, fieldnames=SENTENCE_SELECTION_OUTPUT_HEADER)
        writer.writeheader()
        for user, item, reviewID, aspects in tqdm(
            zip(target["reviewerID"], target["asin"], target["id"], target["aspects"]),
            total=len(target),
        ):
            try:
                start_time = time.time()
                demand = Counter(aspects)
                candidates, demand = get_candidates(
                    corpus, user, item, demand, simplify=True
                )
                if "contextualized" in args.strategy:
                    candidates = contextualize_candidate_sentences(
                        candidates, user, contextualizer
                    )

                if args.debug:
                    candidates_path = os.path.join(
                        args.debug_path, "{}-{}.csv".format(user, item)
                    )
                    if args.verbose:
                        print("Export candidates to {}".format(candidates_path))
                    candidates.to_csv(candidates_path, index=False)

                result = generator.generate(user, item, demand, candidates)

                total_time = time.time() - start_time
                selected_sentences = result.get("selected_sentences", [])
                selected_reviews = result.get("selected_reviews", [])
                record = {
                    "reviewerID": user,
                    "asin": item,
                    "id": reviewID,
                    "selected sentences": selected_sentences,
                    "sentences": " . ".join(selected_sentences),
                    "aspects": array2string(result.get("selected_aspects")),
                    "demand": ",".join(
                        [
                            "{}={}".format(aspect, count)
                            for aspect, count in result.get("demand").items()
                        ]
                    ),
                    "selected reviews": selected_reviews,
                    "n_review": len(set(result["candidates"]["id"])),
                    "n_sentence": len(result["candidates"]),
                    "n_selected_reviews": len(selected_reviews),
                    "n_selected_sentences": len(selected_sentences),
                    "solve time": result.get("solve_time"),
                    "total time": total_time,
                    "objective value": result.get("objective_value"),
                    "objective bound": result.get("objective_bound"),
                    "objective gap": result.get("objective_gap"),
                }

                writer.writerow(record)
                if args.verbose:
                    print(
                        "Done export for user %s item %s" % (user, item),
                        "result",
                        record,
                    )
            except Exception:
                print(
                    "Error occured when generating explanation for user %s item %s"
                    % (user, item)
                )
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print("*** print_tb:")
                traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
                print("*** print_exception:")
                traceback.print_exception(
                    exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout
                )

    print("Done")


if __name__ == "__main__":
    generate_explanations(parse_arguments())
