import os
import json
import sys
import pickle
import string, re
import torch
import random
import openai
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from numpy.linalg import norm
from datasets import Dataset, Features, Sequence, Value, ClassLabel
from collections import Counter, defaultdict, namedtuple, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise
from transformers import RobertaTokenizer, RobertaModel, AutoModelForCausalLM
from rank_bm25 import BM25Okapi
from torch.utils.data import Dataset
from openai.embeddings_utils import get_embedding
from dataset import DataProcessor
from calc_influence_function import calc_influence_single
import utils

openai.api_key = "OPEN_AI.KEY"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    # Initialize variables based on command-line arguments
    embed_type = args.embed
    data = args.data


    # Depending on the dataset chosen, specify the labels and data processor
    if data == "SciERC":
        labels = [
            "Task",
            "OtherScientificTerm",
            "Material",
            "Generic",
            "Method",
            "Metric",
        ]
        dataprocessor = DataProcessor(labels)

        # Load and process training and test data for SciERC dataset
        train_sentences, train_labels, train_ner = dataprocessor.get_data_scierc(
            args.train_file
        )
        test_sentences, test_labels, test_ner = dataprocessor.get_data_scierc(
            args.test_file
        )
    elif data == "ADE":
        labels = ["Adverse-Effect", "Drug"]
        dataprocessor = DataProcessor(labels)
        
        # Load and process training and test data for ADE dataset
        train_sentences, train_labels, train_ners = dataprocessor.get_data_ade(
            args.train_file
        )
        train_sentences, dev_sentences, train_labels, dev_labels = train_test_split(
            train_sentences, train_labels, test_size=0.2
        )
        test_sentences, test_labels = dataprocessor.get_data_ade(args.test_file)
    elif data == "WLPC":
        labels = [
            "Action",
            "Amount",
            "Concentration",
            "Device",
            "Generic-Measure",
            "Location",
            "Measure-Type",
            "Mention",
            "Method",
            "Modifier",
            "Numerical",
            "Reagent",
            "Seal",
            "Size",
            "Speed",
            "Temperature",
            "Time",
            "pH",
        ]
        dataprocessor = DataProcessor(labels)
        
        # Load and process training and test data for WLPC dataset
        train_sentences, train_labels, train_ners = dataprocessor.get_data_wlpc(
            args.train_file
        )
        test_sentences, test_labels, test_ners = dataprocessor.get_data_wlpc(
            args.test_file
        )
    elif data == "STEM-ECR":
        labels = ["Data", "Material", "Method", "Process"]
        dataprocessor = DataProcessor(labels)
        
        # Load and process training and test data for STEM-ECR dataset
        train_sentences, train_labels, train_ners = dataprocessor.get_data_stem(
            args.train_file
        )
        train_sentences, dev_sentences, train_labels, dev_labels = train_test_split(
            train_sentences, train_labels, test_size=0.2
        )
        dev_sentences, test_sentences, dev_labels, test_labels = train_test_split(
            dev_sentences, dev_labels, test_size=0.5
        )

    elif data == "MeasEval":
        labels = ["Quantity"]
        data_dir = "data_files"

        train_docs, train_annotations = os.path.join(
            data_dir, "train/text"
        ), os.path.join(data_dir, "train/tsv")
        
        dev_docs, dev_annotations = os.path.join(data_dir, "trial/txt"), os.path.join(
            data_dir, "trial/tsv"
        )
        test_docs, test_annotations = os.path.join(data_dir, "eval/text"), os.path.join(
            data_dir, "eval/tsv"
        )

        data_processor = DataProcessor(train_docs)
        
        # Load and process training and test data for MeasEval dataset
        train_sentences, train_labels, train_ners = data_processor.get_data_measeval(
            train_docs, train_annotations, part="train"
        )
        dev_sentences, dev_labels, dev_ners = data_processor.get_data_measeval(
            dev_docs, dev_annotations, part="dev"
        )
        test_sentences, test_labels, test_ners = data_processor.get_data_measeval(
            test_docs, test_annotations, part="test"
        )
    else:
        print("Wrong data type")

    # Depending on the model type, load the specified model and tokenizer
    if args.model == "roberta":
        model_path = "roberta-base"
        tok = RobertaTokenizer.from_pretrained(model_path)
        if args.trained:
            model = RobertaModel.from_pretrained(
                args.data + "/results/trained_0roberta-base"
            )
        else:
            model = RobertaModel.from_pretrained(model_path)

        model.to(device)
        X = decode(embed_type, tok, model, train_sentences + test_sentences)
    elif args.model == "api":
        # If the model is based on an API, generate embeddings via the API and load them from a CSV file
        if not os.path.isfile(args.data + "_api.csv"):
            utils.API_decode(
                train_sentences + test_sentences, "text-embedding-ada-002", args.data
            )
        X = pd.read_csv(args.data + "_api.csv")
        
    # Split the dataset into training and development (dev) sets
    n_dev = len(test_sentences)
    n_train = len(train_sentences)

    emb_train = X[n_dev:]
    emb_dev = X[:n_dev]
    
    # Perform nearest neighbor search with different metrics
    for num_neighbors in [1, 3, 5, 10, 20]:
        if args.metric == "euclidean":
            # Use Euclidean distance for nearest neighbor search
            nbrs = NearestNeighbors(
                n_neighbors=num_neighbors, algorithm="ball_tree", n_jobs=-1
            ).fit(emb_train)
            distances, indices = nbrs.kneighbors(emb_dev)
        elif args.metric == "cosine":
            # Use cosine similarity for nearest neighbor search
            dist_matrix = pairwise.cosine_similarity(X=emb_dev, Y=emb_train)
            values, indices = torch.topk(
                torch.from_numpy(dist_matrix), k=num_neighbors, dim=-1
            )
            indices = indices.numpy()
        elif args.metric == "bm25":
            # Use bm25 for icl samples
            tokenized_corpus = [doc.split(" ") for doc in train_sentences]
            bm25 = BM25Okapi(tokenized_corpus)
            sentences = bm25.get_top_n(
                sentence.split(), train_sentences, n=args.num_neighbors
            )
            indices = [
                train_sentences.index(sentences[i]) for i in range(len(sentences))
            ]
        elif args.metric == "influence":
            features = Features(
                {
                    "tokens": Sequence(
                        feature=Value(dtype="string", id=None), length=-1, id=None
                    ),
                    "ner_tags": Sequence(
                        feature=ClassLabel(names=dataprocessor.labels, id=None),
                        length=-1,
                        id=None,
                    ),
                }
            )
            train_ds = Dataset.from_dict(
                {"tokens": train_sentences, "ner_tags": train_labels}, features=features
            )

            dev_ds = Dataset.from_dict(
                {"tokens": dev_sentences, "ner_tags": dev_labels}, features=features
            )

            test_ds = Dataset.from_dict(
                {"tokens": test_sentences, "ner_tags": test_labels}, features=features
            )

            tags = train_ds.features["ner_tags"].feature
            index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
            tag2index = {tag: idx for idx, tag in enumerate(tags.names)}

            tokenized_train_ds = tokenize_and_align_labels(tok, train_ds)
            tokenized_train_ds = tokenized_train_ds.map(
                align_data, batched=True, batch_size=32
            )

            tokenized_dev_ds = tokenize_and_align_labels(tok, dev_ds)
            tokenized_dev_ds = tokenized_dev_ds.map(
                align_data, batched=True, batch_size=32
            )

            tokenized_test_ds = tokenize_and_align_labels(tok, test_ds)
            tokenized_test_ds = tokenized_test_ds.map(
                align_data, batched=True, batch_size=32
            )

            train_loader = DataLoader(tokenized_train_ds, batch_size=32)
            test_loader = DataLoader(tokenized_test_ds, batch_size=32)

            helpful_sentences = []
            harmful_sentenecs = []
            all_train_idxs = list(range(len(train_sentences)))
            for test_idx, test_sentence in enumerate(test_sentences):
                influence, harmful, helpful, _ = calc_influence_single(
                    model,
                    train_loader,
                    test_loader,
                    test_id_num=test_idx,
                    gpu=0,
                    recursion_depth=len(train_sentences),
                    r=len(train_sentences),
                )

        elif args.metric == "perplexity":
            if args.trained:
                model = AutoModelForCausalLM.from_pretrained(
                    args.data + "/results/trained_0roberta-base"
                ).to(device)
            else:
                model = AutoModelForCausalLM.from_pretrained("roberta-base").to(device)
            per = utils.perplexity(tok, model, train_sentences)

            values, indices = torch.topk(-torch.Tensor(per), k=num_neighbors)

        train_indices_np = np.asarray(train_sentences)
        kNN_dev_train = [
            train_indices_np[indices[i]].reshape(1, -1) for i in range(len(indices))
        ]
        kNN_dev_train = np.concatenate(kNN_dev_train, axis=0)

        # Create a data dictionary to store the nearest neighbor results
        if not args.trained:
            PIK = (
                args.data
                + "/kNN/"
                + "{}_{}_{}.dat".format(args.metric, args.embed, num_neighbors)
            )
        else:
            PIK = (
                args.data
                + "/kNN/"
                + "{}_{}_{}_{}_trained.dat".format(
                    args.metric, args.embed, args.model, num_neighbors
                )
            )
        data = dict()
        data["kNN_dev_train"] = kNN_dev_train
        # Save the nearest neighbor results to a file
        with open(PIK, "wb") as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="ADE", type=str)
    parser.add_argument("-m", "--metric", default="euclidean", type=str)
    parser.add_argument("-e", "--embed", default="CLS", type=str)
    parser.add_argument("--model", default="api", type=str)
    parser.add_argument("--trained", default=True, action="store_true")
    parser.add_argument("--reversed", default=False, action="store_true")
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--test_file", type=str)
    parsed_args = parser.parse_args()
    
    # Call the main function with the parsed arguments
    main(parsed_args)
