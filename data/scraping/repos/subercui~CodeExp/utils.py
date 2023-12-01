import os
import json
import itertools
from collections import Counter
from numbers import Number
import operator
from functools import lru_cache
from typing import (
    Iterable,
    List,
    Dict,
    Literal,
    Optional,
    Tuple,
    Union,
)
import numpy as np

import pandas as pd
import openai
from IPython import get_ipython
from rouge import Rouge
import nltk
from nltk import word_tokenize
from nltk.translate.bleu_score import *
from nltk.translate import meteor
from bert_score import BERTScorer

# merge jsonl files
def merge_jsonl_files(files: List[str], output_file: str, overwrite: bool = False):
    # check file extension
    if not output_file.endswith(".jsonl"):
        raise ValueError("output_file should be a jsonl file")
    for f in files:
        if not f.endswith(".jsonl"):
            raise ValueError("files should be jsonl files")
    # check if output file exists
    if os.path.exists(output_file) and not overwrite:
        raise FileExistsError("output file already exists")
    # merge
    with open(output_file, "w") as f:
        for fname in files:
            with open(fname, "r") as f2:
                for line in f2:
                    f.write(line)


# load key in jsonl file
def load_key_jsonl(data_file: str, key: str) -> List[str]:
    if data_file.endswith(".jsonl"):
        with open(data_file) as f:
            data = [json.loads(line) for line in f]
        return [d[key] for d in data]
    else:
        raise ValueError("data_file should be a jsonl file")


# read jsonl file
def load_code_doc(data_file: str) -> Tuple[List[str], Optional[List[str]], List[str]]:
    assert os.path.exists(data_file)
    if data_file.endswith(".jsonl"):
        with open(data_file) as f:
            data = [json.loads(line) for line in f]
        if "docstring" in data[0]:
            docs = [d["docstring"] for d in data]
        else:
            docs = [d["doc"] for d in data]
        block_codes = (
            [d["blocks_codes"] for d in data] if "blocks_codes" in data[0] else None
        )
        codes = [d["code"] for d in data]
    elif data_file.endswith(".json"):
        with open(data_file) as f:
            data = json.load(f)
        docs = [d["docstring"] for d in data]
        block_codes = None
        codes = [d["code"] for d in data]
    return docs, block_codes, codes


# check whether excuting in jupyter notebook
def isnotebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return True  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


# codex call
openai.api_key = os.getenv("OPENAI_API_KEY")


def call_codex(
    code: str,
    mode: Literal["Py2NL", "Py2Doc"],
    max_tokens: int = 256,
) -> Optional[str]:
    """Call codex API with given code"""
    if mode == "Py2NL":
        # example from https://beta.openai.com/examples/default-python-to-natural-language
        prompt = "# Python 3 \n" + code + "\n\n# Explanation of what the code does\n\n#"
        stop = ["Python 3 ", "def "]
    elif mode == "Py2Doc":
        # example from https://beta.openai.com/examples/default-python-docstring
        prompt = (
            "# Python 3.7\n \n"
            + code
            + '\n    \n# An elaborate, high quality docstring for the above function:\n"""'
        )
        stop = ["#", '"""']
    else:
        raise ValueError("mode should be Py2NL or Py2Doc")
    response = openai.Completion.create(
        engine="code-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop,
    )
    if response["choices"][0]["text"]:
        return response["choices"][0]["text"]
    else:
        return "NA"


# functions related to evaluations
def compute_human_ranks(scores: Tuple[float, float], threshold=1e-6) -> Tuple[int, int]:
    """Given a pair of human direct assessment(DA) scores,
       computes the relative ranking. If the difference between
       the two scores is less than the provided threshold,
       the rank is the same.
    Args:
        scores (Tuple[int, int]): A tuple containing the 2 DA scores.
        threshold (int, optional): The threshold of the difference between two scores at which the
                                   the difference is considered (significant). Defaults to 25.
    Returns:
        Tuple[int, int]: The relative ranking of the provided scores
    """
    assert len(scores) == 2
    a, b = scores

    if (a == b) or abs(a - b) < threshold:
        return [1, 1]

    if a > b:
        return [1, 2]

    return [2, 1]


def get_index_pairs(df):
    pairs = list(itertools.combinations(df.index, 2))
    return [pair for pair in pairs if pair[0] != pair[1]]


def compute_rank_pair_type(
    human_ranking: Iterable[Union[int, float]],
    metric_ranking: Iterable[Union[int, float]],
) -> Union[Literal["c"], Literal["d"], Literal["t"], Literal["-"]]:
    """
    Given a pair of human and metric rankings,
    computes the relative ranking. If the difference between
    the two rankings is less than the provided threshold,
    the rank is the same.
    Args:
         human_ranking (Iterable[Union[int, float]]): A pair of human rankings.
         metric_ranking (Iterable[Union[int, float]]): A pair of metric rankings.
    Returns:
         Union[Literal["c"], Literal["d"], Literal["t"], Literal["-"]]: The relative
         ranking of the provided rankings
    """
    comparison_variant_b = [
        (operator.lt, operator.lt, "c"),
        (operator.lt, operator.eq, "t"),
        (operator.lt, operator.gt, "d"),
        (operator.gt, operator.lt, "d"),
        (operator.gt, operator.eq, "t"),
        (operator.gt, operator.gt, "c"),
    ]

    comparison_variant_c = [
        (operator.lt, operator.lt, "c"),
        (operator.lt, operator.eq, "d"),
        (operator.lt, operator.gt, "d"),
        (operator.gt, operator.lt, "d"),
        (operator.gt, operator.eq, "d"),
        (operator.gt, operator.gt, "c"),
    ]

    comparison_variant_d = [
        (operator.lt, operator.lt, "c"),  # <, <
        (operator.lt, operator.eq, "t"),  # <, =
        (operator.lt, operator.gt, "d"),  # <, >
        (operator.eq, operator.lt, "t"),  # =, <
        (operator.eq, operator.eq, "c"),
        (operator.eq, operator.gt, "t"),  # =, >
        (operator.gt, operator.lt, "d"),
        (operator.gt, operator.eq, "t"),
        (operator.gt, operator.gt, "c"),
    ]
    comparison_table = comparison_variant_b

    for h_op, m_op, outcome in comparison_table:
        if h_op(*human_ranking) and m_op(*metric_ranking):
            return outcome

    return "-"


# compute kendall's tau
def kendalls_tau(
    df: pd.DataFrame, human_col: str, metric_col: str, threshold: Number = 0.0001
) -> Tuple[float, Number, Number, Number]:
    counts = Counter()

    pairs = get_index_pairs(df)
    pair_types = []

    for pair in pairs:
        pair_df = df.loc[
            pair,
        ]
        human_scores = pair_df[human_col]
        metric_scores = pair_df[metric_col]

        human_ranks = compute_human_ranks(human_scores, threshold=threshold)
        metric_ranks = metric_scores.rank(method="max", ascending=False)

        pair_type = compute_rank_pair_type(human_ranks, metric_ranks)
        pair_types.append(pair_type)
    counts.update(pair_types)
    concordant_pairs = counts["c"]
    discordant_pairs = counts["d"]
    ties = counts["t"]
    tau = (concordant_pairs - discordant_pairs) / (
        concordant_pairs + discordant_pairs + ties
    )
    print(f"{metric_col} tau: {tau}")
    return tau, concordant_pairs, discordant_pairs, ties


def fast_kendalls_tau(
    df: pd.DataFrame,
    human_col: str,
    metric_col: str,
    threshold: Number = 0.0001,
    remove_invalid: bool = True,
) -> Tuple[float, Number, Number, Number]:

    pairs = get_index_pairs(df)
    left, right = [p[0] for p in pairs], [p[1] for p in pairs]
    left_df, right_df = df.loc[left], df.loc[right]
    if remove_invalid:
        valid_mask = (
            (left_df[metric_col].to_numpy() != 0).astype(int)
            * (right_df[metric_col].to_numpy() != 0).astype(int)
        ) > 0
        left_df = left_df[valid_mask]
        right_df = right_df[valid_mask]
    human_ranks = left_df[human_col].to_numpy(dtype=float) - right_df[
        human_col
    ].to_numpy(dtype=float)
    human_ranks = np.sign(human_ranks)
    metric_ranks = left_df[metric_col].to_numpy(dtype=float) - right_df[
        metric_col
    ].to_numpy(dtype=float)
    metric_ranks = np.sign(metric_ranks)

    non_zero_mask = human_ranks != 0
    # for the non zero subset
    human_ranks = human_ranks[non_zero_mask]
    metric_ranks = metric_ranks[non_zero_mask]

    concordant_pairs = (human_ranks * metric_ranks > 0).sum()
    discordant_pairs = (human_ranks * metric_ranks < 0).sum()
    ties = (metric_ranks == 0).sum()
    tau = (concordant_pairs - discordant_pairs) / (
        concordant_pairs + discordant_pairs + ties
    )
    return tau, concordant_pairs, discordant_pairs, ties


def rouge_score(
    gens,
    docs,
    level: Literal["corpus", "sentence"] = "corpus",
    variant="rouge-1",  # pick the variant when level is sentence
    **kwargs,
):
    rouge = Rouge()
    reference = docs

    if level == "sentence":
        rouges = rouge.get_scores(gens, reference, avg=False)
        # TODO: select rouge-1 or rouge-l and others
        scores = [r[variant]["f"] for r in rouges]
        return scores

    rouges_overall = rouge.get_scores(gens, reference, avg=True)

    print("rouge avg_score: ", rouges_overall)
    return rouges_overall


def rouge_1f(*args, **kwargs):
    kwargs.pop("variant", None)
    return rouge_score(*args, variant="rouge-1", **kwargs)


def rouge_lf(*args, **kwargs):
    kwargs.pop("variant", None)
    return rouge_score(*args, variant="rouge-l", **kwargs)


def nltk_bleu(
    hypotheses: Iterable[str],
    references: Iterable[str],
    level: Literal["corpus", "sentence"] = "corpus",
    **kwargs,
):
    """
    for usage of nltk bleu, see https://www.nltk.org/howto/bleu.html.
    """
    count = 0
    total_score = []

    cc = SmoothingFunction()

    for hyp, ref in zip(hypotheses, references):
        hyp = word_tokenize(hyp)
        ref = word_tokenize(ref)

        try:
            score = nltk.translate.bleu([ref], hyp, smoothing_function=cc.method4)
        except:
            print(hyp, ref)
            score = 0
            raise Exception("bleu error")
        # score = sentence_bleu(ref, hyp, smoothing_function=cc.method4)
        total_score.append(score)
        count += 1

    if level != "corpus":
        return total_score

    avg_score = sum(total_score) / count
    print("blue avg_score: %.4f" % avg_score)
    return avg_score


def meteor_score(
    hypotheses: Iterable[str],
    references: Iterable[str],
    level: Literal["corpus", "sentence"] = "corpus",
    **kwargs,
):
    count = 0
    total_score = []

    for hyp, ref in zip(hypotheses, references):

        score = meteor([word_tokenize(ref)], word_tokenize(hyp))
        # score = sentence_bleu(ref, hyp, smoothing_function=cc.method4)
        total_score.append(score)
        count += 1

    if level != "corpus":
        return total_score

    avg_score = sum(total_score) / count
    print("meteor avg_score: %.4f" % avg_score)
    return avg_score


@lru_cache(maxsize=16)
def _load_scorer(
    scorer_name: Literal["BERT", "codeBERT"],
) -> BERTScorer:
    if scorer_name == "BERT":
        return BERTScorer(lang="en")
    elif scorer_name == "codeBERT":
        return BERTScorer(
            model_type="microsoft/codebert-base",
            num_layers=9,
            lang="en",
        )
    else:
        raise ValueError(f"unknown scorer: {scorer_name}")


def bertscore(hypotheses, references, level="corpus", **kwargs):
    bert_scorer = _load_scorer("BERT")
    count = 0
    total_score = []

    for hyp, ref in zip(hypotheses, references):
        P, R, F1 = bert_scorer.score([hyp], [ref])
        total_score.append(float(F1[0]))
        count += 1

    if level != "corpus":
        return total_score

    avg_score = sum(total_score) / count
    print("bertscore avg_score: %.4f" % avg_score)
    return avg_score


def codebertscore(hypotheses, references, level="corpus", **kwargs):
    codebert_scorer = _load_scorer("codeBERT")
    count = 0
    total_score = []

    for hyp, ref in zip(hypotheses, references):
        P, R, F1 = codebert_scorer.score([hyp], [ref])
        total_score.append(float(F1[0]))
        count += 1

    if level != "corpus":
        return total_score

    avg_score = sum(total_score) / count
    print("codebert avg_score: %.4f" % avg_score)
    return avg_score


def fact_scores(
    gens,
    docs,
    codes,
    level="corpus",
    mode="recall",
    **kwargs,
) -> Union[List, float]:
    def calc_recall(common_1_grams, gen_doc):
        if len(common_1_grams) == 0:
            return 0
        gen_1_grams = set(word_tokenize(gen_doc))
        return len(gen_1_grams.intersection(common_1_grams)) / len(common_1_grams)

    def calc_f1(common_1_grams, gen_doc):
        if len(common_1_grams) == 0 or len(word_tokenize(gen_doc)) == 0:
            return 0
        gen_1_grams = set(word_tokenize(gen_doc))
        recall = len(gen_1_grams.intersection(common_1_grams)) / len(common_1_grams)
        precision = len(gen_1_grams.intersection(common_1_grams)) / len(gen_1_grams)
        if precision + recall == 0:
            return 0
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    if mode == "f1":
        calc_score = calc_f1
    else:
        calc_score = calc_recall
    fact_scores = []
    for i in range(len(codes)):
        code = codes[i]
        gen_doc = gens[i]
        doc = docs[i]
        common_1_grams = set(word_tokenize(code)) & set(word_tokenize(doc))
        fact_scores.append(calc_score(common_1_grams, gen_doc))

    mean_score = sum(fact_scores) / len(fact_scores)
    print(f"fact_score: {mean_score}")

    if level != "corpus":
        return fact_scores
    return mean_score


class Evaluator:
    def __init__(self, eval_fn, **kwargs):
        self.eval_fn = eval_fn
        self.kwargs = kwargs

    def __call__(self, gens, docs, codes, level="corpus", **kwargs):
        return self.eval_fn(gens, docs, codes=codes, level=level, **kwargs)

    def eval_df(self, df, code_col, ref_col, gen_col, out_col=None, **kwargs):
        """
        evaluate on a dataframe.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas dataframe")
        if code_col not in df.columns:
            raise ValueError(f"code_col {code_col} not in df")
        if ref_col not in df.columns:
            raise ValueError(f"ref_col {ref_col} not in df")
        if gen_col not in df.columns:
            raise ValueError(f"gen_col {gen_col} not in df")
        if out_col:
            if out_col in df.columns:
                raise ValueError(f"out_col {out_col} already in df")

        result = self(
            df[gen_col].tolist(),
            df[ref_col].tolist(),
            df[code_col].tolist(),
            level="sentence",
            **kwargs,
        )
        if out_col:
            df[out_col] = result
        return result
