import os
import re
import string
from collections import defaultdict
from itertools import compress
from json import dumps
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from hdbscan import HDBSCAN
from matplotlib.colors import ListedColormap
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import InvertedRBO, TopicDiversity
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from umap import UMAP

dirname = os.path.dirname(__file__)
outputs_dir = os.path.join(dirname, "artifacts")

from utils import (
    BERTopic,
    CTMScikit,
    Doc2VecScikit,
    LatentDirichletAllocation,
    SentenceTransformerScikit,
    Timer,
)

VALID_EMBEDDINGS_MODELS = [
    "doc2vec",
    "sentence-transformers/msmarco-distilbert-base-v4",
]
VALID_TOPIC_MODELS = ["BERTopic", "CTM", "LDA"]
UMAP_EVAL_K_RANGE = [10, 20, 40, 80, 160]
N_CV_SPLITS = 5

# Explicitly disable parallelism to avoid any hidden deadlock
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# TODO:
# - Partition the data into train and test set. Do the hyperparameter tuning on
# the train set only and evaluate the best model on the test set.
# - Use other datasets for validating the methodology
# - Use other embedding models (e.g. TF-IDF)
# - Simplify the validation mechanism (maybe use simple train-test split) to make the experiments achievable
# - Define MLflow project file


def clean_text(text):
    """Cleans a string of text by applying common transformations.

    Args:
        text (string): a string of text.

    Returns:
        string: a clean string of text.
    """
    re_url = re.compile(
        r"(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
    )
    re_email = re.compile(
        "(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"
    )
    text = text.lower()
    text = text.strip()
    text = re.sub(re_url, "", text)
    text = re.sub(re_email, "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"(\d+)", " ", text)
    text = re.sub(r"(\s+)", " ", text)

    return text


def prepare_20newsgroups(dataset_file=None):
    """Applies the necessary pre-processing to the 20newsgroups corpus.

    Args:
        dataset_file (Path | string, optional): the path to the csv file where the pre-processed corpus should
        be saved. Defaults to None.

    Returns:
        list(list, list, list): the transformed documents, respective target values and target labels (ordered) .
    """
    print("Load and clean the dataset.")
    y_labels = [
        "alt.atheism",
        "comp.graphics",
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
        "comp.windows.x",
        "misc.forsale",
        "rec.autos",
        "rec.motorcycles",
        "rec.sport.baseball",
        "rec.sport.hockey",
        "sci.crypt",
        "sci.electronics",
        "sci.med",
        "sci.space",
        "soc.religion.christian",
        "talk.politics.guns",
        "talk.politics.mideast",
        "talk.politics.misc",
        "talk.religion.misc",
    ]

    if dataset_file:
        # Check whether there is a saved dataset in disk
        if os.path.isfile(dataset_file):
            # Load the data from disk
            df = pd.read_csv(dataset_file)
            X_clean, y_clean = df["X_clean"], df["y_clean"]

            return X_clean.tolist(), y_clean.tolist(), y_labels

    # Loading the data
    newsgroups_data = fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes")
    )
    X, y = newsgroups_data.data, newsgroups_data.target

    # Clean text
    X_clean = list(map(clean_text, X))
    blank = np.array([len(doc) > 2 for doc in X_clean])  # Remove blank documents
    fourwords = np.array(
        [len(doc.split(" ")) > 4 for doc in X_clean]
    )  # Remove documents with 4 words or less
    outliers = np.array(
        ["the most current orbital" not in doc for doc in X_clean]
    )  # Remove outliers (in embedding space)
    X_clean = list(compress(X_clean, blank & fourwords & outliers))
    y_clean = list(compress(y, blank & fourwords & outliers))

    # Save the dataset to disk
    if dataset_file:
        pd.DataFrame({"X_clean": X_clean, "y_clean": y_clean}).to_csv(
            dataset_file, index=False
        )

    return X_clean, y_clean, y_labels


def suggest_hyperparameters(trial):
    """Optuna suggest hyperparameters function. Given an Optuna study trial, sample the hyperparameter according
    to the pre-defined sampler.

    Args:
        trial (optuna.trial.Trial): an Optuna trial object.

    Raises:
        ValueError: when the suggested topic_model hyperparameter is not defined.

    Returns:
        dict: a dictionary with the sampled hyperparameters.
    """
    print("Suggest hyperparameters.")
    hyperparams = {}

    # Define embedding_model
    hyperparams["embedding_model"] = trial.suggest_categorical(
        "embedding_model", VALID_EMBEDDINGS_MODELS
    )
    if hyperparams["embedding_model"] == "doc2vec":
        hyperparams["dm"] = trial.suggest_categorical("dm", [0, 1])

    # Define UMAP hyperparameters
    hyperparams["umap_args__n_neighbors"] = trial.suggest_int("n_neighbors", 10, 50)
    hyperparams["umap_args__n_components"] = trial.suggest_categorical(
        "n_components", [2, 5, 10, 25, 50]
    )
    hyperparams["umap_args__metric"] = trial.suggest_categorical(
        "metric", ["cosine", "euclidean"]
    )

    # Define topics_model
    hyperparams["topic_model"] = trial.suggest_categorical(
        "topic_model", VALID_TOPIC_MODELS
    )
    if hyperparams["topic_model"] == "BERTopic":
        # Setting hyperparameters for BERTopic
        hyperparams["min_topic_size"] = trial.suggest_int("min_topic_size", 10, 60)
        hyperparams["hdbscan_args__min_cluster_size"] = trial.suggest_int(
            "min_cluster_size", 30, 150
        )
        hyperparams["hdbscan_args__cluster_selection_epsilon"] = trial.suggest_float(
            "cluster_selection_epsilon", 0.01, 1.0, log=True
        )
        hyperparams[
            "hdbscan_args__cluster_selection_method"
        ] = trial.suggest_categorical("cluster_selection_method", ["eom", "leaf"])

    elif hyperparams["topic_model"] == "CTM":
        # Setting hyperparameters for CTM
        hyperparams["model_type"] = trial.suggest_categorical(
            "model_type", ["prodLDA", "LDA"]
        )
        hyperparams["activation"] = trial.suggest_categorical(
            "activation", ["relu", "softplus"]
        )
        hyperparams["hidden_sizes"] = trial.suggest_categorical(
            "hidden_sizes",
            [(100,), (100, 100), (100, 100, 100), (300,), (300, 300), (300, 300, 300)],
        )
        hyperparams["num_epochs"] = trial.suggest_categorical(
            "num_epochs", [33, 66, 100]
        )
        hyperparams["dropout"] = trial.suggest_float("dropout", 0.0, 0.4)
        hyperparams["lr"] = trial.suggest_float("lr", 2e-3, 2e-1)
        hyperparams["inference_type"] = trial.suggest_categorical(
            "inference_type", ["zeroshot", "combined"]
        )

    elif hyperparams["topic_model"] == "LDA":
        # Setting hyperparameters for LDA
        hyperparams["learning_decay"] = trial.suggest_float("learning_decay", 0.5, 1.0)
        hyperparams["max_iter"] = trial.suggest_int("max_iter", 2, 10)
        hyperparams["max_doc_update_iter"] = trial.suggest_int(
            "max_doc_update_iter", 50, 200
        )

    else:
        raise ValueError(f"topic_model={hyperparams['topic_model']} is not defined!")

    # Log parameters with mlflow
    mlflow.log_params(trial.params)

    # Print the hyperparemeters
    print(dumps(hyperparams, sort_keys=False, indent=2))

    return hyperparams


def define_embedding_model(hyperparams):
    """Instantiate the Embedding Model based on the sampled embedding_model hyperparameter.

    Args:
        hyperparams (dict): a dictionary with the sampled hyperparameters.

    Raises:
        ValueError: when the suggested embedding_model hyperparameter is not defined.

    Returns:
        Doc2VecScikit | SentenceTransformerScikit: an instance of the sampled Embedding Model.
    """
    print("Define the embedding model.")
    embedding_model = hyperparams["embedding_model"]
    if embedding_model == "doc2vec":
        # Based on hyperparameters used in Top2Vec
        model = Doc2VecScikit(
            dm=hyperparams["dm"],
            dbow_words=1,
            vector_size=300,
            min_count=50,
            window=15,
            sample=1e-5,
            negative=0,
            hs=1,
            epochs=40,
            seed=0,
            workers=cpu_count() - 1,
        )

    elif embedding_model == "sentence-transformers/msmarco-distilbert-base-v4":
        model = SentenceTransformerScikit(
            model_name_or_path=embedding_model,
            show_progress_bar=True,
            cache_folder=outputs_dir,
        )

    else:
        raise ValueError(f"embedding_model={embedding_model} is not defined!")

    return model


def define_topic_model(hyperparams):
    """Instantiate the Topic Model based on the sampled topic_model hyperparameter.

    Args:
        hyperparams (dict): a dictionary with the sampled hyperparameters.

    Raises:
        ValueError: when the suggested topic_model hyperparameter is not defined.

    Returns:
        BERTopic | CTMScikit | LatentDirichletAllocation: an instance of the sampled Topic Model.
    """
    print("Define the topic model.")
    if hyperparams["topic_model"] == "BERTopic":
        # Setting model components
        umap_model = UMAP(
            n_neighbors=hyperparams["umap_args__n_neighbors"],
            n_components=hyperparams["umap_args__n_components"],
            min_dist=0.0,
            metric=hyperparams["umap_args__metric"],
            random_state=1,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=hyperparams["hdbscan_args__min_cluster_size"],
            cluster_selection_epsilon=hyperparams[
                "hdbscan_args__cluster_selection_epsilon"
            ],
            cluster_selection_method=hyperparams[
                "hdbscan_args__cluster_selection_method"
            ],
            metric="euclidean",
            memory=outputs_dir,  # we cache the hard computation and recompute only the relatively cheap flat cluster extraction
            prediction_data=True,
        )
        vectorizer_model = CountVectorizer(ngram_range=(1, 1), stop_words="english")
        n_gram_range = (1, 1)

        # Declaring the model
        model = BERTopic(
            top_n_words=10,
            n_gram_range=n_gram_range,
            nr_topics=20,
            min_topic_size=hyperparams["min_topic_size"],
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
        )

    elif hyperparams["topic_model"] == "CTM":
        # Declaring the model
        model = CTMScikit(
            n_components=20,
            model_type=hyperparams["model_type"],
            activation=hyperparams["activation"],
            hidden_sizes=hyperparams["hidden_sizes"],
            num_epochs=hyperparams["num_epochs"],
            dropout=hyperparams["dropout"],
            lr=hyperparams["lr"],
            inference_type=hyperparams["inference_type"],
            num_data_loader_workers=cpu_count() - 1,
        )

    elif hyperparams["topic_model"] == "LDA":
        # Declaring the model
        model = LatentDirichletAllocation(
            learning_decay=hyperparams["learning_decay"],
            max_iter=hyperparams["max_iter"],
            max_doc_update_iter=hyperparams["max_doc_update_iter"],
            n_components=20,
            random_state=0,
        )

    else:
        raise ValueError(f"topic_model={hyperparams['topic_model']} is not defined!")

    return model


def evaluate_umap(umap_emb_train, umap_emb_test, y_train, y_test):
    """Evaluate UMAP projections using K-NN Classifier accuracy over multiple K.
    For each K, fit a K-NN classifier on the UMAP embeddings and classify each document.
    Evaluate the classifications using the Accuracy metric.

    Args:
        umap_emb_train (np.ndarray): the UMAP embeddings of the training documents.
        umap_emb_test (np.ndarray): the UMAP embeddings of the testing documents.
        y_train (np.ndarray): the target values of the training documents.
        y_test (np.ndarray): the target values of the testing documents.

    Returns:
        list(dict, dict): the training and testing accuracies.
    """
    accuracies_train = {}
    accuracies_test = {}
    for k in UMAP_EVAL_K_RANGE:
        # Initialize the KNN classifier
        knn = KNeighborsClassifier(
            n_neighbors=k,
            weights="uniform",
            algorithm="brute",
            metric="cosine",
        )

        # Get KNN classifier predictions
        # Cast to float32 to avoid ValueError("Input contains NaN, infinity or a value too large for dtype('float32')
        umap_emb_train = np.float32(umap_emb_train)
        knn.fit(umap_emb_train, y_train)
        y_train_pred = knn.predict(umap_emb_train)
        y_test_pred = knn.predict(umap_emb_test)

        # Compute accuracies
        accuracies_train[f"umap_{k}nn_acc_train"] = accuracy_score(
            y_train, y_train_pred
        )
        accuracies_test[f"umap_{k}nn_acc_test"] = accuracy_score(y_test, y_test_pred)

    return accuracies_train, accuracies_test


def evaluate_cluster(topics, y, outlier_label=None):
    """Compare cluster labels with target values using Normalized Mutual Information (NMI).
    NMI is a scaled (between 0 and 1) version of Mutual Information (MI) and it quantifies the
    "amount of information obtained about one random varible by observing the other random variable".
    Ideally, for a good cluster solution that captures the target labels, NMI should be close to 1.

    Args:
        topics (np.ndarray): the topic assignments to each document.
        y (np.ndarray): the target values of each document.
        outlier_label (int, optional): a label that indicates documents that are outliers.
        If passed, two NMI values will be computed, one with outliers and another removing outliers.
        Defaults to None.

    Returns:
        list(float, float | None): the NMI value(s).
    """
    assert len(topics) == len(
        y
    ), f"topics and y have different lengths ({len(topics)}, {len(y)})."
    nmi = normalized_mutual_info_score(y, topics)
    if outlier_label:
        doc_ids = [top != outlier_label for top in topics]
        nmi_filtered = normalized_mutual_info_score(
            list(compress(y, doc_ids)), list(compress(topics, doc_ids))
        )
        return nmi, nmi_filtered
    else:
        return nmi, None


def evaluate_topic(model_output, texts):
    """Evaluate the topics provided by the topic model according to Diversity and Coherence.
    Use three metrics to evaluate the topics Topic Diversity, Inverted Ranked-Biased Overlap, and
    Topic Coherence.

    Args:
        model_output (dict): a dict with three keys: 'topic', 'topic-word-matrix', and 'topic-document-matrix'.
        texts (list): a list of lists, representing the vectorized training corpus.

    Returns:
        list(float, float, float): the Topic Diversity, Inverted Ranked-Biased Overlap, and
        Topic Coherence values.
    """
    # Topic Diversity
    metric = TopicDiversity(topk=10)
    topic_diversity = metric.score(model_output)

    # Topic Diversity - inverted RBO
    metric = InvertedRBO(topk=10)
    inverted_rbo = metric.score(model_output)

    # Topic Coherence
    metric = Coherence(texts=texts, topk=10, measure="c_v")
    topic_coherence_c_v = metric.score(model_output)

    return topic_diversity, inverted_rbo, topic_coherence_c_v


def plot_umap_labels(umap_emb, y, y_labels, topics, topic_labels):
    """Create a matplotlib.pyplot.Figure with the UMAP projection of the documents and the
    respective original and topic labels. Allows the visual comparison between the documents'
    original and topic labels.

    Args:
        umap_emb (np.ndarray): the UMAP embeddings of the documents.
        y (np.ndarray): the target values of each document.
        y_labels (list): the target labels (ordered).
        topics (np.ndarray): the topic assignments to each document.
        topic_labels (list): the topic labels (ordered).

    Returns:
        list(matplotlib.pyplot.Figure, matplotlib.pyplot.Figure): matplotlib.pyplot.Figure objects
        with scatter plots of UMAP original and topic labels.
    """
    # Plot the 2D UMAP projection with the topic labels vs original labels
    # Figure 1 - Original labels
    fig1, ax1 = plt.subplots(figsize=(11, 9))
    s0 = ax1.scatter(umap_emb[:, 0], umap_emb[:, 1], s=4, c=y, cmap="tab20")
    ax1.legend(
        s0.legend_elements(num=len(y_labels))[0],
        y_labels,
        bbox_to_anchor=(1, 1),
        loc="upper left",
        fontsize=11,
        framealpha=1,
    )

    # Figure 2 - Topic labels
    fig2, ax2 = plt.subplots(figsize=(11, 9))
    outcmap = ListedColormap(
        [(0.8509803921568627, 0.8509803921568627, 0.8509803921568627)]
    )
    s1 = ax2.scatter(
        umap_emb[topics == -1, 0],
        umap_emb[topics == -1, 1],
        s=2,
        c=topics[topics == -1],
        marker=".",
        cmap=outcmap,
    )
    s1handles, _ = s1.legend_elements(num=1)
    s2 = ax2.scatter(
        umap_emb[topics != -1, 0],
        umap_emb[topics != -1, 1],
        s=6,
        c=topics[topics != -1],
        cmap="tab20",
    )
    s2handles, _ = s2.legend_elements(num=len(topic_labels) - 1)
    ax2.legend(
        s1handles + s2handles,
        topic_labels,
        bbox_to_anchor=(1, 1),
        loc="upper left",
        fontsize=11,
        framealpha=1,
    )

    return fig1, fig2


def train_infer_models(topic_model, umap_model, emb_model, X_train, X_test):
    """Train the topic, umap and embedding models on the training data and use them to Infer
    the testing data.

    Args:
        topic_model (BERTopic | CTMScikit | LatentDirichletAllocation): an instance of the sampled Topic Model.
        umap_model (UMAP): an instance of the UMAP model.
        emb_model (Doc2VecScikit | SentenceTransformerScikit): an instance of the sampled Embedding Model.
        X_train (np.ndarray): the training corpus.
        X_test (np.ndarray): the testing corpus.

    Returns:
        dict: all the models' outputs necessary for evaluation.
    """
    infer = {}

    # Initialize timers
    timer_emb_model_t = Timer("timer_emb_model_train")
    timer_emb_model_i = Timer("timer_emb_model_infer")
    timer_top_model_t = Timer("timer_top_model_train")
    timer_top_model_i = Timer("timer_top_model_infer")

    # Fit and transform the embedding model
    print(f"Fit and transform the {emb_model} embedding model.")
    timer_emb_model_t.start()
    emb_train = emb_model.fit_transform(X_train)
    timer_emb_model_t.stop()
    timer_emb_model_i.start()
    emb_test = emb_model.transform(X_test)
    timer_emb_model_i.stop()

    # Fit and transform the topic model
    print(f"Fit and transform the {topic_model} topic model.")
    timer_top_model_t.start()
    infer["top_train"] = topic_model.fit_transform(X_train, embeddings=emb_train)
    timer_top_model_t.stop()
    timer_top_model_i.start()
    infer["top_test"] = topic_model.transform(X_test, embeddings=emb_test)
    timer_top_model_i.stop()

    # Fit and transform the UMAP model on 2 components
    print(f"Reduce embeddings to 2 dimensions with UMAP.")
    infer["umap_emb_train"] = umap_model.fit_transform(emb_train)
    infer["umap_emb_test"] = umap_model.transform(emb_test)

    # Get full output dictionary from topic_model
    infer["tm_full_output"] = topic_model.full_output

    # Add timers to infer dictionary
    infer["timers"] = Timer.timers

    return infer


def evaluate_models(infer, y_train, y_test, X_train, y_labels=None, last_iter=False):
    """Evaluate the topic, umap and embedding models using the evaluate_umap, evaluate_cluster,
    evaluate_topic, and plot_umap_labels functions.

    Args:
        infer (dict): all the models' outputs necessary for evaluation.
        y_train (np.ndarray): the training corpus target values.
        y_test (np.ndarray): the testing corpus.
        X_train (np.ndarray): the training corpus.
        y_labels (list, optional): the target labels (ordered). Defaults to None.
        last_iter (bool, optional): indicates if it is the last iteration of the Cross-Validation process.
        Defaults to False.

    Returns:
        dicts: all the evaluation metrics to be logged.
    """
    artifacts = {}

    # Save the number of topics identified
    artifacts["ntopics"] = len(infer["tm_full_output"]["topics"])

    # Save the percentage of observations classified as outliers (outliers should be labeled as -1)
    artifacts["perc_outliers_train"] = (infer["top_train"] == -1).mean()
    artifacts["perc_outliers_test"] = (infer["top_test"] == -1).mean()

    # Evaluate the UMAP model on test split
    print("Evaluate UMAP on K-NN accuracy.")
    knn_accuracies_train, knn_accuracies_test = evaluate_umap(
        infer["umap_emb_train"], infer["umap_emb_test"], y_train, y_test
    )
    artifacts.update(knn_accuracies_train)
    artifacts.update(knn_accuracies_test)

    # Evaluate the clustering on agreement between true labels and topics
    # 0 value indicates two independent label assignments; 1 value indicates two agreeable label assignments
    print("Evaluate clustering on Mutual Information.")
    artifacts["nmi_train"], artifacts["nmi_filtered_train"] = evaluate_cluster(
        infer["top_train"], y_train, outlier_label=-1
    )
    artifacts["nmi_test"], artifacts["nmi_filtered_test"] = evaluate_cluster(
        infer["top_test"], y_test, outlier_label=-1
    )

    # Evaluate the Topic model on Coherence and Diversity metrics
    print("Evaluate topics on Diversity and Coherence metrics.")
    (
        artifacts["topic_diversity"],
        artifacts["inverted_rbo"],
        artifacts["topic_coherence_c_v"],
    ) = evaluate_topic(
        infer["tm_full_output"], list(map(lambda x: x.split(" "), X_train))
    )

    if last_iter:
        # Add timers to artifacts
        artifacts.update(infer["timers"])

        # UMAP plot on last split of k-fold cross validation: Original labels VS Topics
        print("Produce UMAP plot: Original labels VS Topics.")
        topic_names = [
            ".".join(words[:5]) for words in infer["tm_full_output"]["topics"]
        ]
        topic_names[0] = "outliers"  # setting the outlier topic name
        artifacts["orig_train_fig"], artifacts["top_train_fig"] = plot_umap_labels(
            infer["umap_emb_train"], y_train, y_labels, infer["top_train"], topic_names
        )
        artifacts["orig_test_fig"], artifacts["top_test_fig"] = plot_umap_labels(
            infer["umap_emb_test"], y_test, y_labels, infer["top_test"], topic_names
        )

    return artifacts


def objective(trial):
    """The Optuna objective function. This function is used by Optuna to quantify the performance
    of the chosen hyperparameters and to guide future hyperparameter sampling.
    This function returns three objectives: umap_knn_acc_test_mean (average over UMAP_EVAL_K_RANGE),
    nmi_filtered_test_mean, and topic_coherence_c_v_mean.
    Additionaly, this function logs every parameter, metric and artifact necessary for comparing
    the models a posteriori.

    Args:
        trial (optuna.trial.Trial): an Optuna trial object.

    Returns:
        list(float, float, float): the objective values to optimize.
    """
    with mlflow.start_run():
        # Load dataset
        X_clean, y_clean, y_labels = prepare_20newsgroups(
            os.path.join(outputs_dir, "20newsgroups_prep.csv")
        )

        # Suggest hyperparameters
        hyperparams = suggest_hyperparameters(trial)

        # Define embedding model
        emb_model = define_embedding_model(hyperparams)

        # Define topic model
        topic_model = define_topic_model(hyperparams)

        # Define UMAP model for projecting space to 2 dimensions
        umap_model = UMAP(
            n_neighbors=hyperparams["umap_args__n_neighbors"],
            n_components=2,
            min_dist=0.0,
            metric=hyperparams["umap_args__metric"],
            random_state=1,
        )

        # Apply Stratified K-fold
        split_metrics = defaultdict(list)
        skf = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=1)
        mlflow.log_param("cv-folds", N_CV_SPLITS)
        for n, (train_ix, test_ix) in enumerate(skf.split(X_clean, y_clean)):
            print(f"Iteration number {n + 1} out of {N_CV_SPLITS}.")

            # Get train and test samples
            X_train, X_test = np.array(X_clean)[train_ix], np.array(X_clean)[test_ix]
            y_train, y_test = np.array(y_clean)[train_ix], np.array(y_clean)[test_ix]

            # Train and infer with topic_model, umap_model and emb_model
            infer = train_infer_models(
                topic_model, umap_model, emb_model, X_train, X_test
            )

            # Evaluate the topic_model and umap_model
            if n == N_CV_SPLITS - 1:
                artifacts = evaluate_models(
                    infer, y_train, y_test, X_train, y_labels, last_iter=True
                )
            else:
                artifacts = evaluate_models(infer, y_train, y_test, X_train)

            # Append split metrics
            for k, v in artifacts.items():
                if "fig" in k:
                    continue
                elif "timer" in k:
                    split_metrics[k] = v
                else:
                    split_metrics[k].append(v)

        print("Log artifacts.")
        # Get averages and standard deviations of metrics
        agg_metrics = {}
        for k, v in split_metrics.items():
            agg_metrics[k + "_mean"] = np.mean(v)
            agg_metrics[k + "_std"] = np.std(v)

        # Get average of umap_knn_acc_test_means and standard deviations
        agg_metrics["umap_avgknn_acc_test_mean"] = np.mean(
            [agg_metrics[f"umap_{k}nn_acc_test_mean"] for k in UMAP_EVAL_K_RANGE]
        )
        agg_metrics["umap_avgknn_acc_test_std"] = np.mean(
            [agg_metrics[f"umap_{k}nn_acc_test_std"] for k in UMAP_EVAL_K_RANGE]
        )

        # Log metrics with mlflow
        mlflow.log_metrics(agg_metrics)

        # Log figures with mlflow
        mlflow.log_figure(artifacts["orig_train_fig"], "orig_umap_train_plot.png")
        mlflow.log_figure(artifacts["top_train_fig"], "top_umap_train_plot.png")
        mlflow.log_figure(artifacts["orig_test_fig"], "orig_umap_test_plot.png")
        mlflow.log_figure(artifacts["top_test_fig"], "top_umap_test_plot.png")

        # Get evaluation metric(s)
        eval_metrics = [
            agg_metrics["umap_avgknn_acc_test_mean"],  # UMAP eval metric
            agg_metrics["nmi_filtered_test_mean"],  # Cluster eval metric
            agg_metrics["topic_coherence_c_v_mean"],  # Topic modeling eval metric
        ]

        return eval_metrics


def _get_best_results(best_trial):
    """Used for obtaining the data necessary to produce the figures offsite."""
    with mlflow.start_run(run_name="best-model"):
        # Load dataset
        X_clean, y_clean, y_labels = prepare_20newsgroups(
            os.path.join(outputs_dir, "20newsgroups_prep.csv")
        )

        # Suggest hyperparameters
        hyperparams = suggest_hyperparameters(best_trial)

        # Define embedding model
        emb_model = define_embedding_model(hyperparams)

        # Define topic model
        topic_model = define_topic_model(hyperparams)

        # Define UMAP model for projecting space to 2 dimensions
        umap_model = UMAP(
            n_neighbors=hyperparams["umap_args__n_neighbors"],
            n_components=2,
            min_dist=0.0,
            metric=hyperparams["umap_args__metric"],
            random_state=1,
        )

        # Get train and test samples
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=0, stratify=y_clean
        )

        # Train and infer with topic_model, umap_model and emb_model
        infer = train_infer_models(topic_model, umap_model, emb_model, X_train, X_test)

        return X_train, X_test, y_train, y_test, y_labels, infer


def log_best_model(best_trial):
    """Re-evaluate the model with the best hyperparameters on a train/ test split.

    Args:
        best_trial (optuna.FrozenTrial): the Optuna best trial object.
    """
    with mlflow.start_run(run_name="best-model"):
        # Load dataset
        X_clean, y_clean, y_labels = prepare_20newsgroups(
            os.path.join(outputs_dir, "20newsgroups_prep.csv")
        )

        # Suggest hyperparameters
        hyperparams = suggest_hyperparameters(best_trial)

        # Define embedding model
        emb_model = define_embedding_model(hyperparams)

        # Define topic model
        topic_model = define_topic_model(hyperparams)

        # Define UMAP model for projecting space to 2 dimensions
        umap_model = UMAP(
            n_neighbors=hyperparams["umap_args__n_neighbors"],
            n_components=2,
            min_dist=0.0,
            metric=hyperparams["umap_args__metric"],
            random_state=1,
        )

        # Get train and test samples
        mlflow.log_param("test_size", 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=0, stratify=y_clean
        )

        # Train and infer with topic_model, umap_model and emb_model
        infer = train_infer_models(topic_model, umap_model, emb_model, X_train, X_test)

        # Evaluate the topic_model and umap_model
        artifacts = evaluate_models(
            infer, y_train, y_test, X_train, y_labels, last_iter=True
        )

        print("Log artifacts.")
        # Get average of umap_knn_acc_test
        artifacts["umap_avgknn_acc_test"] = np.mean(
            [artifacts[f"umap_{k}nn_acc_test"] for k in UMAP_EVAL_K_RANGE]
        )

        # Extract single elements from timer artifacts
        for k, v in artifacts.items():
            if "timer" in k:
                artifacts[k] = v[0]

        # Log metrics with mlflow
        orig_train_fig = artifacts.pop("orig_train_fig")
        top_train_fig = artifacts.pop("top_train_fig")
        orig_test_fig = artifacts.pop("orig_test_fig")
        top_test_fig = artifacts.pop("top_test_fig")
        mlflow.log_metrics(artifacts)

        # Log figures with mlflow - if the figures get cut, use fig.savefig(path, bbox_inches='tight')
        mlflow.log_figure(orig_train_fig, "orig_umap_train_plot.png")
        mlflow.log_figure(top_train_fig, "top_umap_train_plot.png")
        mlflow.log_figure(orig_test_fig, "orig_umap_test_plot.png")
        mlflow.log_figure(top_test_fig, "top_umap_test_plot.png")


if __name__ == "__main__":
    # Hyperparameter tuning
    print("Performing Hyper-parameter tuning.")
    mlflow.set_tracking_uri(os.path.join(outputs_dir, "mlruns"))
    mlflow.set_experiment("mapintel-experiment")
    study = optuna.create_study(
        directions=["maximize", "maximize", "maximize"]
    )  # maximize the 3 evaluation metrics
    print(
        f"Starting optimization process! Sampler is {study.sampler.__class__.__name__}"
    )
    study.optimize(
        objective, n_trials=100, n_jobs=1
    )  # Set gc_after_trial=True if you see an increase in memory consumption over trials

    # Print optuna study statistics
    print("\n++++++++++++++++++++++++++++++++++\n")
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trials:")
    best_trials = study.best_trials

    for trial in best_trials:
        print("  Trial number: ", trial.number)
        print("  Loss (trial value): ", trial.value)

        print("  Params: ")
        print(dumps(trial.params, sort_keys=False, indent=2))

        print("  Log optimal model: ")
        log_best_model(trial)
