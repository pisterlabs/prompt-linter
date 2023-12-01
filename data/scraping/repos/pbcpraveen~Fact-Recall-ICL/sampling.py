import os
import json
import requests
import argparse
import openai
from dotenv import load_dotenv
from tqdm import tqdm
from nltk import word_tokenize
import string
import pdb
from tqdm import tqdm
import pandas as pd
import numpy as np
from datasets import load_dataset
import re
import pickle
from utils import *
import time

COUNT = 200
load_dotenv('api_key.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

print("loading and sampling data points...")

wikibio_dataset = load_dataset("wiki_bio", split="train")
sample_points = sample_random_people(COUNT, wikibio_dataset)


sample_points.to_csv("result/sample3.csv")







