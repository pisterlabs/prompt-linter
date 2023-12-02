"""
Script to train a model to
topical modeling of customer content reviews
"""
import os
import pickle
from argparse import ArgumentParser

import mlflow
from gensim.models import CoherenceModel, LdaModel
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner


@task
def load_dataset(input_dir: str) -> tuple[list, list]:
    with open(os.path.join(input_dir, "train_corpus.pkl"), "rb") as f_in:
        train_dataset = pickle.load(f_in)

    with open(os.path.join(input_dir, "valid_tokens.pkl"), "rb") as f_in:
        val_dataset = pickle.load(f_in)

    return train_dataset, val_dataset


@task
def load_indexed_dictionary(input_dir: str) -> list:
    with open(os.path.join(input_dir, "id2word.pkl"), "rb") as f_in:
        return pickle.load(f_in)


def compute_coherence_values(X_train, X_val, passes, id2word, k, a, b) -> float:
    lda_model = LdaModel(
        corpus=X_train,
        id2word=id2word,
        num_topics=k,
        random_state=123,
        chunksize=100,
        passes=passes,
        alpha=a,
        eta=b,
    )

    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=X_val, dictionary=id2word, coherence="c_v"
    )

    return coherence_model_lda.get_coherence()


@task
def hyperparameter_opt(
    X_train: list,
    X_val: list,
    id2word: list,
    passes: int,
    topics: range,
    alpha: list,
    beta: list,
):
    with mlflow.start_run():
        mlflow.set_tag("model", "LDA")
        mlflow.set_tag("scope", "Topic-Modeling")
        mlflow.log_param("passes", passes)

        for k in topics:
            for a in alpha:
                for b in beta:
                    params = {}
                    params["k"] = k
                    params["a"] = a
                    params["b"] = b

                    coherence_score = compute_coherence_values(
                        X_train, X_val, passes, id2word, k, a, b
                    )

                    mlflow.log_metric("coherence", coherence_score)
                    mlflow.log_metrics(params)


# pylint: disable=line-too-long
@flow(
    name="topic-modeling-training-pipeline-params-opt",
    task_runner=SequentialTaskRunner(),
)
def train(
    experiment_name: str,
    input_dir: str,
    passes: int,
    topics: list,
    alpha: list,
    beta: list,
):
    topics = range(topics[0], topics[1], topics[2])
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(experiment_name)

    train_dataset, val_dataset = load_dataset(input_dir)

    id2word = load_indexed_dictionary(input_dir)

    hyperparameter_opt(train_dataset, val_dataset, id2word, passes, topics, alpha, beta)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment to be used for tracking",
        required=True,
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        help="Data's path to be used for model training",
        required=True,
    )

    parser.add_argument(
        "--passes",
        type=int,
        help="Number of passes to be used for model training",
        default=3,
    )

    parser.add_argument(
        "--topics",
        type=list,
        help=(
            "Number of topics to be used for model training.            Expected three values:"
            " start, stop, step"
        ),
        default=[5, 20, 5],
    )

    parser.add_argument(
        "--alpha",
        type=list,
        help="Alpha hyperparameter to be used for model training",
        default=[0.01, 0.31, 0.61, 0.91],
    )

    parser.add_argument(
        "--beta",
        type=list,
        help="Beta hyperparameter to be used for model training",
        default=[0.01, 0.31, 0.61, 0.91],
    )

    args = parser.parse_args()

    train(
        args.experiment_name,
        args.input_dir,
        args.passes,
        args.topics,
        args.alpha,
        args.beta,
    )
