from rlhfutils.eval_utils import oai_kwargs, load_alldfs, annotate_apfarm, apf_format, load_wgpt, filter_and_sort_df
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt
import re
from transformers import AutoTokenizer
from datasets import load_dataset
import openai
from rlhfutils.data import qaform

SFT_MODEL_PATH = ""
GENERATED_OUTPUT_FOLDER = "generated_outs/"
toker = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
adfs = load_alldfs(GENERATED_OUTPUT_FOLDER,  500)

print(adfs.keys())

# original thing to compare with simulated preferences
ORIGNAME = "stackorigrerun"
# list of keys to compare against ORIGNAME with APFarmEval  
trykeys = [ 'stacklenonlyppo3']

assert len(adfs[trykeys[0]])>400

for t in trykeys:
    assert t in adfs.keys()

for k in trykeys:
    print(len(adfs[k]))
    # match everything against original PPO
    lenannot = annotate_apfarm(adfs, k, ORIGNAME, 100, len(adfs[k]), oai_kwargs())