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
from collections import Counter, defaultdict, namedtuple, defaultdict
from sklearn.model_selection import train_test_split
from dataset import DataProcessor
import utils

openai.api_key = "OPENAI.KEY"


def main(args):
    embed_type = args.embed
    data = args.data
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

        train_sentences, train_labels, train_ner = dataprocessor.get_data_scierc(
            args.train_file
        )
        test_sentences, test_labels, test_ner = dataprocessor.get_data_scierc(
            args.test_file
        )
    elif data == "ADE":
        labels = ["Adverse-Effect", "Drug"]
        dataprocessor = DataProcessor(labels)
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
        train_sentences, train_labels, train_ners = dataprocessor.get_data_wlpc(
            args.train_file
        )
        test_sentences, test_labels, test_ners = dataprocessor.get_data_wlpc(
            args.test_file
        )
    elif data == "STEM-ECR":
        labels = ["Data", "Material", "Method", "Process"]
        dataprocessor = DataProcessor(labels)
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

    if os.path.isfile(args.icl_file_name):
        with open(args.icl_file_name, "rb") as f:
            data = pickle.load(f)

        token_level_f1, entity_level_f1 = utils.evaluation(
            test_sentences[0:100], test_labels[0:100], data, dataprocessor
        )
    else:
        print("No exist icl sample file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="ADE", type=str)
    parser.add_argument("--icl_file_name", default="euclidean", type=str)
    parser.add_argument("--trained", default=True, action="store_true")
    parser.add_argument("--reversed", default=False, action="store_true")
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--test_file", type=str)
    parsed_args = parser.parse_args()
    main(parsed_args)
