import logging
import os
import sys

import numpy as np

sys.path.append("..")
sys.path.append("../utils")
import argparse
import json
import math
import os
import pickle
import sys
import tarfile
import time
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import regex as re
import requests
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, CLIPModel, CLIPProcessor
from utils.metrics_hai import compute_metrics
from utils.utils import *
from utils.utils import loss_01

# from mmlu_aux import *
logging.getLogger().setLevel(logging.INFO)


class MMLUDataset:
    def __init__(
        self,
        data_x,
        expl_embs,
        questions,
        answers,
        to_shows,
        data_y,
        ai_preds,
        ai_scores,
        ai_expl,
        metadata,
        metadata_labels,
        subjects,
        metric_y,
        ai_model,
    ):
        self.data_x = data_x
        self.expl_embs = expl_embs
        self.data_y = np.array(data_y, dtype="int")
        self.ai_preds = np.array(ai_preds, dtype="int")
        self.ai_scores = np.array(ai_scores)
        self.ai_expl = np.array(ai_expl)
        self.to_shows = np.array(to_shows)
        self.questions = questions
        self.answers = answers
        self.metadata = metadata
        self.metadata_labels = metadata_labels  # category labels
        self.metric_y = metric_y
        self.subjects = subjects
        self.ai_model = ai_model

    def print_example(self, idx, to_print=False):
        str_to_print = ""
        str_to_print += f"Question: {self.questions[idx]}"
        str_to_print += f"A1: {self.answers[idx][0]}"
        str_to_print += f"A2: {self.answers[idx][1]}"
        str_to_print += f"A3: {self.answers[idx][2]}"
        str_to_print += f"A4: {self.answers[idx][3]}"
        str_to_print += (
            f"Correct Answer: {self.data_y[idx]}- {self.answers[idx][self.data_y[idx]]}"
        )
        if to_print:
            logging.info(str_to_print)
        return str_to_print

    def __len__(self):
        return len(self.data_y)


def download_mmlu(path_data):
    data_path = path_data + "/mmlu"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
        r = requests.get(url, allow_redirects=True)
        with open(data_path + "/data.tar", "wb") as f:
            f.write(r.content)
        # untar
        tar = tarfile.open(data_path + "/data.tar")
        tar.extractall(data_path)
        tar.close()
        # remove tar
        print(f"Removing data.tar")
        os.remove(data_path + "/data.tar")


def prepare_mmlu_dataset(path_data, model_llm="openai", n_gpu=1, key_openai="YOUR_KEY"):
    """
    model_llm: openai(gpt-3.5-turbo) or any hf model e.g. google/flan-t5-small or google/flan-t5-xl"
    """
    download_mmlu(path_data)
    if model_llm == "openai":
        openai.api_key = key_openai
        eval_model_openai(
            "openai3", path_data + "/mmlu/data", path_data + "/mmlu/exp_data"
        )
    else:
        eval_model_hf(
            model_llm, path_data + "/mmlu/data", path_data + "/mmlu/exp_data", n_gpu
        )

    questions = []
    answers = []
    ai_preds = []
    ai_scores = []
    data_y = []
    metadatas = []
    ai_expls = []
    to_shows = []
    metadata_labels = ["subject", "subcat", "cat"]
    path_test = path_data + "/mmlu/data/test"
    if model_llm == "openai":
        path_model_preds = path_data + "/mmlu/exp_data/results_openai3"
    else:
        path_model_preds = path_data + "/mmlu/exp_data/results_" + model_llm

    subjects = sorted(
        [f.split(".csv")[0] for f in os.listdir(path_model_preds) if ".csv" in f]
    )

    for subject in subjects:
        answers_df = pd.read_csv(os.path.join(path_model_preds, subject + ".csv"))

        for i in range(len(answers_df)):
            question = answers_df.iloc[i, 0]
            answer = [
                answers_df.iloc[i, 1],
                answers_df.iloc[i, 2],
                answers_df.iloc[i, 3],
                answers_df.iloc[i, 4],
            ]
            if model_llm == "openai":
                ai_score = 0
            else:
                ai_score = [
                    float(answers_df.iloc[i, 7]),
                    float(answers_df.iloc[i, 8]),
                    float(answers_df.iloc[i, 9]),
                    float(answers_df.iloc[i, 10]),
                ]

            str_to_print = ""
            str_to_print += f"Question: {question} <br>"
            str_to_print += f"A: {answer[0]} <br>"
            str_to_print += f"B: {answer[1]} <br>"
            str_to_print += f"C: {answer[2]} <br>"
            str_to_print += f"D: {answer[3]} <br>"

            to_shows.append(str_to_print)
            # ai pred is the index of the highest score
            # label = test_df.iloc[i, 5]
            label = answers_df.iloc[i, 5]
            ai_pred = answers_df.iloc[i, -1][0]
            ai_pred = ord(ai_pred) - ord("A") + 1
            ai_expl = answers_df.iloc[i, -2]
            # covert from A, B, C, D to 0, 1, 2, 3
            label = ord(label) - ord("A") + 1
            metadata = [subject, subcategories[subject][0]]
            # find the category
            for cat, subcat_lists in categories.items():
                if subcategories[subject][0] in subcat_lists:
                    metadata.append(cat)
                    break
            questions.append(question)
            answers.append(answer)
            ai_scores.append(ai_score)
            ai_preds.append(ai_pred)
            ai_expls.append(ai_expl)
            data_y.append(label)
            metadatas.append(metadata)

    """     #OBQA

    answers_df =  pd.read_csv(
        os.path.join(path_model_preds +"o",  "obqa.csv"))

    for i in range(len(answers_df)):
        question = answers_df.iloc[i, 0]
        answer = [answers_df.iloc[i, 1], answers_df.iloc[i, 2], answers_df.iloc[i, 3], answers_df.iloc[i, 4]]
        ai_score = 0#[float(answers_df.iloc[i,7]), float(answers_df.iloc[i,8]), float(answers_df.iloc[i,9]), float(answers_df.iloc[i,10])]

        str_to_print = ''
        str_to_print += f'Question: {question} <br>'
        str_to_print += f'A: {answer[0]} <br>'
        str_to_print += f'B: {answer[1]} <br>'
        str_to_print += f'C: {answer[2]} <br>'
        str_to_print += f'D: {answer[3]} <br>'

        to_shows.append(str_to_print)
        # ai pred is the index of the highest score
        #label = test_df.iloc[i, 5]
        label = answers_df.iloc[i,5]
        ai_pred = answers_df.iloc[i,-1][0]
        ai_pred = ord(ai_pred) - ord("A") + 1 
        ai_expl = answers_df.iloc[i,-2]
        # covert from A, B, C, D to 0, 1, 2, 3
        label = ord(label) - ord("A") + 1
        metadata = ["obqa", "other","other"]
        # find the category

        questions.append(question)
        answers.append(answer)
        ai_scores.append(ai_score)
        ai_preds.append(ai_pred)
        ai_expls.append(ai_expl)
        data_y.append(label)
        metadatas.append(metadata)
            
    """

    # convert to numpy
    questions = np.array(questions)
    answers = np.array(answers)
    ai_scores = np.array(ai_scores)
    ai_preds = np.array(ai_preds)
    ai_expls = np.array(ai_expls)
    data_y = np.array(data_y)
    metadatas = np.array(metadatas)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # batch size of 16
    embeddings = model.encode(to_shows, batch_size=16)
    embeddings = np.array(embeddings)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sentences = [ai_expls[i] for i in range(len(ai_expls))]
    embeddings_exp = model.encode(sentences, batch_size=16)
    embeddings_exp = np.array(embeddings_exp)

    dataset_mmlu = MMLUDataset(
        embeddings,
        embeddings_exp,
        questions,
        answers,
        to_shows,
        data_y,
        ai_preds,
        ai_scores,
        ai_expls,
        metadatas,
        metadata_labels,
        subjects,
        loss_01,
        model_llm,
    )

    with open(path_data + "/mmlu/mmlu_dataset_" + model_llm + ".pkl", "wb") as f:
        pickle.dump(dataset_mmlu, f)
    return dataset_mmlu
