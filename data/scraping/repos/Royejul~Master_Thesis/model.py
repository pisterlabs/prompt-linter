# The code used in this document was developed specifically for the thesis titled :
# "Comparative Analysis of Statistical and Machine Learning Methods for Topic Modeling of Research Paper Datasets"
# author = Julien Feuillade
# last version from 03 July 2023

import csv
import os
import time
from typing import Mapping, Any, List, Tuple, Optional

import warnings
from numba.core.errors import NumbaDeprecationWarning

warnings.filterwarnings('ignore', category=NumbaDeprecationWarning)

import numpy as np
import tensorflow_hub as hub
from bertopic import BERTopic
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.models.LDA import LDA
from octis.models.NMF import NMF
from octis.models.NeuralLDA import NeuralLDA
from sentence_transformers import SentenceTransformer
from octis.dataset.dataset import Dataset

from itertools import product


def train_LDA(dataset, params: Mapping[str, any] = None) -> Tuple[Mapping[str, Any], float, int]:
    """
    Trains an LDA model on the given dataset.

    Parameters:
        - dataset (Any): The input dataset from OCTIS.

        - params (Mapping[str, Any]): Optional. A dictionary containing additional parameters for training.
          Default value is None.

    Returns:
        - Tuple[Mapping[str, Any], float, int]: A tuple containing the trained model, the computation time in seconds,
          and the number of topics in the trained model.
    """

    # Test if the parameters is empty or not and create the model
    if params is None:
        model = LDA()
    else:
        model = LDA(**params)
    start = time.time()
    # Train the model
    output = model.train_model(dataset)
    end = time.time()
    computation_time = float(end - start)
    # Get the number of topics
    num_topics = output['topic-word-matrix'].shape[0]
    return output, computation_time, num_topics


def train_NMF(dataset, params: Mapping[str, any] = None) -> Tuple[Mapping[str, Any], float]:
    """
    Trains an NMF model on the given dataset.

    Parameters:
        - dataset (Any): The input dataset from OCTIS.

        - params (Mapping[str, Any]): Optional. A dictionary containing additional parameters for training.
          Default value is None.

    Returns:
        - Tuple[Mapping[str, Any], float, int]: A tuple containing the trained model, the computation time in seconds,
          and the number of topics in the trained model.
    """

    # Test if the parameters is empty or not and create the model
    if params is None:
        model = NMF()
    else:
        model = NMF(**params)
    start = time.time()
    # Train the model
    output = model.train_model(dataset)
    end = time.time()
    computation_time = float(end - start)
    # Get the number of topics
    num_topics = output['topic-word-matrix'].shape[0]
    return output, computation_time, num_topics


def train_NeuralLDA(dataset, params: Mapping[str, any] = None) -> Tuple[Mapping[str, Any], float]:
    """
    Trains an NeuralLDA model on the given dataset.

    Parameters:
        - dataset (Any): The input dataset from OCTIS.

        - params (Mapping[str, Any]): Optional. A dictionary containing additional parameters for training.
          Default value is None.

    Returns:
        - Tuple[Mapping[str, Any], float, int]: A tuple containing the trained model, the computation time in seconds,
          and the number of topics in the trained model.
    """

    # Test if the parameters is empty or not and create the model
    if params is None:
        model = NeuralLDA()
    else:
        model = NeuralLDA(**params)
    start = time.time()
    # Train the model
    output = model.train_model(dataset)
    end = time.time()
    computation_time = float(end - start)
    # Get the number of topics
    num_topics = output['topic-word-matrix'].shape[0]
    return output, computation_time, num_topics


def create_sentence_transformer_embeddings(dataset):
    """
    Creates sentence embeddings using the SentenceTransformer model.

    Parameters:
        - dataset (Any): The input dataset from OCTIS.

    Returns:
        - Tuple[Any, float]: A tuple containing the sentence embeddings and the computation time in seconds.
    """

    start = time.time()
    # Combines words in each element of dataset into a single string with spaces
    data = [" ".join(words) for words in dataset.get_corpus()]
    # Initialize the SentenceTransformer model using the "all-MiniLM-L6-v2" pre-trained model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    # Encode the data using the SentenceTransformer model to obtain sentence embeddings
    embeddings = embedding_model.encode(data)
    end = time.time()
    computation_time = float(end - start)
    return embeddings, computation_time


def create_USE_embeddings(dataset):
    """
    Creates sentence embeddings using the USE model.

    Parameters:
        - dataset (Any): The input dataset from OCTIS.

    Returns:
        - Tuple[Any, float]: A tuple containing the sentence embeddings and the computation time in seconds.
    """

    start = time.time()
    # Combines words in each element of dataset into a single string with spaces
    data = [" ".join(words) for words in dataset.get_corpus()]
    # Load the Universal Sentence Encoder model from TensorFlow Hub
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    # Generate sentence embeddings using the Universal Sentence Encoder model
    embeddings = embed(data).numpy()
    end = time.time()
    computation_time = float(end - start)
    return embeddings, computation_time


def create_doc2vec_embeddings(dataset, vector_size=300, window=5, min_count=1, epochs=10):
    """
    Creates Doc2Vec embeddings using the gensim Doc2Vec model.

    Parameters:
        - dataset (Any): The input dataset from OCTIS.

        - vector_size (int): Optional. The dimensionality of the document embeddings. Default is 300.

        - window (int): Optional. The maximum distance between the predicted word and context words within a document.
          Default is 5.

        - min_count (int): Optional. The minimum count of words to consider when training the model. Default is 1.

        - epochs (int): Optional. The number of iterations (epochs) over the dataset for training. Default is 10.

    Returns:
        - Tuple[np.ndarray, float]: A tuple containing the document embeddings as a NumPy array and the computation time in seconds.
    """

    start = time.time()
    # Convert dataset into a format suitable for Doc2Vec by tagging each document with an index
    tagged_data = [TaggedDocument(words, [i]) for i, words in enumerate(dataset.get_corpus())]
    # Initialize a Doc2Vec model with specified parameters
    model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=4, epochs=epochs)
    # Build the vocabulary of the model using the tagged data
    model.build_vocab(tagged_data)
    # Train the model on the tagged data for the specified number of epochs
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    # Generate document embeddings by inferring vectors for each document in the dataset
    embeddings = [model.infer_vector(words) for words in dataset.get_corpus()]
    end = time.time()
    computation_time = float(end - start)
    return np.array(embeddings), computation_time


def train_BERTopic(dataset, params: Mapping[str, any] = None, embeddings: List[str] = None) -> Tuple[
    Mapping[str, Any], float]:
    """
    Trains a BERTopic model on the given dataset.

    Parameters:
        - dataset (Any): The input dataset from OCTIS.

        - params (Mapping[str, Any]): Optional. A dictionary containing additional parameters for training.
          Default value is None.

        - embeddings (List[str]): Optional. A list of pre-calculated embeddings to be used during training.
          Default value is None.

    Returns:
        - Tuple[Mapping[str, Any], float]: A tuple containing the trained model, the computation time in seconds.
    """

    # Combines words in each element of dataset into a single string with spaces
    data = [" ".join(words) for words in dataset.get_corpus()]
    # Test if the parameters is empty or not and create the model
    if params is None:
        model = BERTopic()
    else:
        model = BERTopic(**params)  # Create model
    start = time.time()
    # Test if the embeddings is empty or not and add the pre-calculate embedding
    if embeddings is not None:
        topics, _ = model.fit_transform(data, embeddings=embeddings)
    else:
        topics, _ = model.fit_transform(data)
    # Make the output of BERTopic compatible with OCTIS
    bertopic_topics = []
    for i in range(-1, len(set(topics))):
        topic = model.get_topic(i)
        if isinstance(topic, bool):
            continue  # Ignore bool values
        keyword = [word[0] for word in topic]
        bertopic_topics.append(keyword)
    output = {"topics": bertopic_topics}
    end = time.time()
    computation_time = float(end - start)
    # Get the number of topics
    num_topics = len(model.get_topic_freq())
    return output, computation_time, num_topics


def get_unique_filename(base_filename):
    """
    Generates a unique filename by appending a version number to the base filename if it already exists.

    Parameters:
        - base_filename (Union[str, os.PathLike]): The base filename or path for generating the unique filename.

    Returns:
        - str: A unique filename with a version number appended to the base filename if it already exists.
    """

    version = 0
    new_filename = base_filename
    while os.path.exists(new_filename):
        version += 1
        new_filename = f"{base_filename[:-4]}_v{version}{base_filename[-4:]}"
    return new_filename


def calculate_coherence_scores(dataset, models_config, topic_numbers=None, output_csv=None):
    """
    Calculates coherence scores for different topic models on the given dataset.

    Parameters:
        - dataset (Any): The input dataset for calculating coherence scores.

        - models_config (List[Dict[str, Any]]): A list of dictionaries representing the configuration of each model.
          Each dictionary should contain "name", "train_func", and optional "embeddings", "embedding_time", and "params".

        - topic_numbers (Optional[List[int]]): Optional. A list of topic numbers to evaluate the models on.
          If None, the default topic numbers from the models_config will be used. Default is None.

        - output_csv (Optional[Union[str, os.PathLike]]): Optional. The output CSV file to write the results to.
          If provided, a unique filename will be generated to avoid overwriting existing files. Default is None.

    Returns:
        - Tuple[float, float]: A tuple containing the final coherence score and the computation time in seconds.
    """

    # Compute corpus coherence once for the entire dataset
    coherence = Coherence(texts=dataset.get_corpus(), topk=10, measure="c_npmi")

    csvfile = None
    writer = None
    try :
        if output_csv is not None:
            output_csv = get_unique_filename(output_csv)
            csvfile = open(output_csv, "w", newline='')
            fieldnames = ["model_name", "topics", "coherence", "time", "time_with_embedding"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        for model_config in models_config:
            model_name = model_config["name"]
            train_func = model_config["train_func"]
            embeddings = model_config.get("embeddings", None)
            embedding_time = model_config.get("embedding_time", None)
            params = model_config.get("params", None)

            if topic_numbers is None:
                topic_numbers = [None]

            for i in topic_numbers:
                if i is not None:
                    if params is None:
                        params = {}
                    if model_name in {"LDA", "NeuralLDA", "NMF"}:
                        params["num_topics"] = i
                    else:
                        params["nr_topics"] = i

                start = time.time()
                if embeddings is not None:
                    output, _, num_topics = train_func(dataset=dataset, params=params, embeddings=embeddings)
                else:
                    output, _, num_topics = train_func(dataset=dataset, params=params)
                coherence_score = coherence.score(output)
                end = time.time()
                computation_time = float(end - start)

                if embedding_time is not None:
                    embedding_time = computation_time + embedding_time
                else:
                    embedding_time = computation_time

                print(f'{model_name} coherence {num_topics} topics: {coherence_score:.4f} ({computation_time:.4f} sec)')

                result = {
                    "model_name": model_name,
                    "topics": num_topics,
                    "coherence": coherence_score,
                    "time": computation_time,
                    "time_with_embedding": embedding_time
                }

                if writer is not None:
                    writer.writerow(result)
    finally:
        if csvfile is not None:
            csvfile.close()

    return coherence_score, computation_time


def calculate_diversity_scores(dataset, models_config, topic_numbers=None, output_csv="diversity_scores.csv"):
    """
    Calculates diversity scores for different topic models on the given dataset.

    Parameters:
        - dataset (Any): The input dataset for calculating coherence scores.

        - models_config (List[Dict[str, Any]]): A list of dictionaries representing the configuration of each model.
          Each dictionary should contain "name", "train_func", and optional "embeddings", "embedding_time", and "params".

        - topic_numbers (Optional[List[int]]): Optional. A list of topic numbers to evaluate the models on.
          If None, the default topic numbers from the models_config will be used. Default is None.

        - output_csv (Optional[Union[str, os.PathLike]]): Optional. The output CSV file to write the results to.
          If provided, a unique filename will be generated to avoid overwriting existing files. Default is None.

    Returns:
        - Tuple[float, float]: A tuple containing the final coherence score and the computation time in seconds.
    """

    output_csv = get_unique_filename(output_csv)
    # Compute corpus coherence once for the entire dataset
    diversity = TopicDiversity(topk=10)

    with open(output_csv, "w", newline='') as csvfile:
        fieldnames = ["model_name", "topics", "diversity", "time", "time_with_embedding"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model_config in models_config:
            model_name = model_config["name"]
            train_func = model_config["train_func"]
            embeddings = model_config.get("embeddings", None)
            embedding_time = model_config.get("embedding_time", None)
            params = model_config.get("params", None)

            if topic_numbers is None:
                topic_numbers = [None]

            for i in topic_numbers:
                if i is not None:
                    if params is None:
                        params = {}
                    if model_name in {"LDA", "CTM", "NeuralLDA", "NMF"}:
                        params["num_topics"] = i
                    else:
                        params["nr_topics"] = i

                start = time.time()
                if embeddings is not None:
                    output, _, num_topics = train_func(dataset=dataset, params=params, embeddings=embeddings)
                else:
                    output, _, num_topics = train_func(dataset=dataset, params=params)
                diversity_score = diversity.score(output)
                end = time.time()
                computation_time = float(end - start)

                if embedding_time is not None:
                    embedding_time = computation_time + embedding_time
                else:
                    embedding_time = computation_time

                print(f'{model_name} diversity {num_topics} topics: {diversity_score:.4f} ({computation_time:.4f} sec)')

                # Write to CSV
                writer.writerow({
                    "model_name": model_name,
                    "topics": num_topics,
                    "diversity": diversity_score,
                    "time": computation_time,
                    "time_with_embedding": embedding_time
                })


def calculate_computation_times(models_config, dataset_sizes, output_csv="computation_times.csv"):
    """
    Calculates the computation times for different models on different dataset sizes.

    Parameters:
        - models_config (List[Dict[str, Any]]): A list of dictionaries representing the configuration of each model.
          Each dictionary should contain "name", "train_func", and optional "embeddings" and "params".

        - dataset_sizes (List[int]): A list of dataset sizes to evaluate the models on.

        - output_csv (Union[str, os.PathLike]): Optional. The output CSV file to write the results to.
          If provided, a unique filename will be generated to avoid overwriting existing files.
          Default is "computation_times.csv".

    Returns:
        - None
    """

    output_csv = get_unique_filename(output_csv)
    with open(output_csv, "w", newline='') as csvfile:
        fieldnames = ["model_name", "dataset_size", "total_documents", "vocabulary_length", "embedding_time",
                      "model_time", "total_time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for dataset_size in dataset_sizes:
            dataset = Dataset()
            dataset.load_custom_dataset_from_folder(f"dataset_octis/research_dataset_{dataset_size}")
            metadata = dataset.get_metadata()

            for model_config in models_config:
                model_name = model_config["name"]
                train_func = model_config["train_func"]
                embeddings_func = model_config.get("embeddings", None)
                params = model_config.get("params", None)

                # Calculate embedding time
                start_embedding = time.time()
                if embeddings_func is not None:
                    preprocessed_embeddings = embeddings_func(dataset)
                else:
                    preprocessed_embeddings = None
                end_embedding = time.time()
                embedding_time = float(end_embedding - start_embedding)

                print(f'{model_name} on dataset {dataset_size}: embedding time: {embedding_time:.4f} sec')

                # Calculate model time
                start_model = time.time()
                if preprocessed_embeddings is not None:
                    _, _, _ = train_func(dataset=dataset, params=params, embeddings=preprocessed_embeddings)
                else:
                    _, _, _ = train_func(dataset=dataset, params=params)
                end_model = time.time()
                model_time = float(end_model - start_model)

                total_time = embedding_time + model_time

                print(
                    f'{model_name} on dataset {dataset_size}: model time: {model_time:.4f} sec, total time: {total_time:.4f} sec')

                # Write to CSV
                writer.writerow({
                    "model_name": model_name,
                    "dataset_size": dataset_size,
                    "total_documents": metadata["total_documents"],
                    "vocabulary_length": metadata["vocabulary_length"],
                    "embedding_time": embedding_time,
                    "model_time": model_time,
                    "total_time": total_time
                })


def hyperparameter_optimization(dataset, param_grid, model_config, output_csv = None):
    """
    Performs hyperparameter optimization by evaluating different parameter combinations on the given dataset.

    Parameters:
        - dataset (Any): The input dataset from OCTIS.

        - param_grid (Dict[str, List[Any]]): A dictionary specifying the hyperparameter grid to search.
          Each key corresponds to a hyperparameter name, and each value is a list of possible values for that hyperparameter.

        - model_config (Dict[str, Any]): A dictionary representing the configuration of the model to optimize.
          It should contain "name", "train_func", and optional "embeddings".

        - output_csv (Optional[Union[str, os.PathLike]]): Optional. The output CSV file to write the results to.
          If provided, a unique filename will be generated to avoid overwriting existing files. Default is None.

    Returns:
        - Tuple[float, Dict[str, Any]]: A tuple containing the best coherence score and the corresponding best parameters.
    """

    csvfile = None
    writer = None

    try:
        if output_csv is not None:
            output_csv = get_unique_filename(output_csv)
            csvfile = open(output_csv, "w", newline='')
            fieldnames = ["params", "coherence"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        all_params = list(product(*param_grid.values()))

        best_coherence = float("-inf")  # we want to maximize coherence
        best_params = None

        for params in all_params:
            params_dict = dict(zip(param_grid.keys(), params))

            # overwrite the params in model_config
            model_config['params'] = params_dict

            models_config = [model_config]  # we want to test only this configuration

            print(params_dict)
            coherence_scores, _ = calculate_coherence_scores(dataset, models_config)

            # assuming coherence_scores is the last coherence score calculated
            coherence = coherence_scores

            if coherence > best_coherence:  # we want to maximize coherence
                best_coherence = coherence
                best_params = params_dict

            result = {
                "params": params_dict,
                "coherence": coherence,
            }

            if writer is not None:
                writer.writerow(result)
    finally:
        if csvfile is not None:
            csvfile.close()

    return best_coherence, best_params
