import guidance
import torch
from model_dict import load_from_catalogue
from guidance_model import Summarization
import argparse
import pandas as pd
import csv
from tqdm import tqdm
import numpy as np
import os
import pdb
import json
import scipy.stats

# score given two np arrays with scipy.stats.kendalltau
def score_kendall(pred, gold):
    return scipy.stats.kendalltau(pred, gold)[0]

# load data from csv file
def load_csv(path):
    data = pd.read_csv(path)
    return data

# load data from json file
def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data