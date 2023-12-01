#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import tomllib
import itertools

import pandas as pd
import tomotopy as tp
from tomotopy.utils import Corpus
from tomotopy.coherence import Coherence

def build_corpus(indir: Path) -> Corpus:
    """Build a Corpus."""
    corpus = Corpus()
    for path in indir.glob("*.txt"):
        with path.open('r') as fin:
            doc = fin.read().split()
            corpus.add_doc(doc)

    return corpus

def model_metrics(
    corpus: Corpus
    , n_topics: int
    , alpha: float
    , eta: float
) -> tuple[float, float]:
    """Build a topic model and retrieve metrics from it."""
    model = tp.LDAModel(
        k = n_topics, alpha = alpha, eta = eta, corpus = corpus, seed = 357
    )
    model.train(iter = 1000)
    coherence_model = Coherence(model, coherence = 'c_v')
    
    return model.perplexity, coherence_model.get_score()

def main(args: argparse.Namespace) -> None:
    """Run the script."""
    # Build the corpus
    corpus = build_corpus(args.indir)

    # Load the hyperparameters TOML
    with args.params.open('rb') as fin:
        params = tomllib.load(fin)

    # Permute the parameters
    k = range(params['kmin'], params['kmax'])
    param_list = list(itertools.product(k, params['alphas'], params['etas']))
    print("Parameter combinations to search:", len(param_list))

    # Grid search
    results = []
    for idx, param_set in enumerate(param_list):
        n_topics, alpha, eta = param_set
        perplexity, coherence = model_metrics(corpus, n_topics, alpha, eta)
        run = {
            'n_topics': n_topics
            , 'alpha': alpha
            , 'eta': eta
            , 'perplexity': perplexity
            , 'coherence': coherence
        }
        results.append(run)

        if idx % 10 == 0:
            print(f"Completed {idx} of {len(param_list)} combinations")

    results = pd.DataFrame(results)
    results.to_csv(args.results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Grid search a list of hyperparameters for topic modeling"
    )
    parser.add_argument('--indir', type=Path, help="Input directory")
    parser.add_argument('--params', type=Path, help="Parameters to search")
    parser.add_argument('--results', type=Path, help="Results manifest")
    args = parser.parse_args()
    main(args)

