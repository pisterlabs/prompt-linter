# Train LDA model.
# -*- coding: utf-8 -*-

"""
This script will be used to preprocess the Guardian Text Data
It is designed to be idempotent [stateless transformation]
Usage:
    python3 ./src/pipeline/train.py
"""

import click
from gensim.models import LdaModel, LdaMulticore
from gensim.models import CoherenceModel

from src.pipeline.preprocess import HtmlSubdirsCorpus
from src.utils import parse_config, set_logger


@click.command()
@click.argument("config_file", type=str, default="src/config.yml")
@click.argument("corpus_directory", type=str, default="./data/raw/body/test_train_py")
def train(config_file, corpus_directory):
    """
    Train LDA model
    Args:
        config_file [str]: path to config file
    Returns:
        corpus [HtmlSubdirsCorpus]: Streamable Corpus
    """

    # Configure Logger
    logger = set_logger("./logs/train.log")

    # Load Config from Config File
    # logger.info(f"Load config from {config_file}")
    config = parse_config(config_file)
    # logger.info(f"config: {config['train']} parsed")

    # Load Corpus lazely
    corpus = HtmlSubdirsCorpus(corpus_directory=corpus_directory, config=config)

    import logging

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    model = LdaMulticore(
        corpus=corpus,
        num_topics=config["train"]["num_topics"],
        chunksize=config["train"]["chunksize"],
        iterations=config["train"]["iterations"],
        passes=config["train"]["passes"],
        eval_every=config["train"]["eval_every"],
    )

    perplexity = model.log_perplexity(corpus)
    logger.info(f"perplexity: {perplexity}")
    coherence_model = CoherenceModel(model=model, corpus=corpus, coherence="u_mass")
    coherence = coherence_model.get_coherence()
    logger.info(f"coherence: {coherence}")

    # for vector in corpus:

    # # Train model
    # model = LdaModel(
    #     corpus=corpus,
    #     id2word=corpus.id2token,
    #     chunksize=config["preprocess"]["chunksize"],
    #     alpha="auto",
    #     eta="auto",
    #     iterations=config["preprocess"]["iterations"],
    #     num_topics=config["preprocess"]["num_topics"],
    #     passes=config["preprocess"]["passes"],
    #     eval_every=config["preprocess"]["eval_every"],
    # )

    # Serialize Model
    # model.save()


if __name__ == "__main__":
    train()
