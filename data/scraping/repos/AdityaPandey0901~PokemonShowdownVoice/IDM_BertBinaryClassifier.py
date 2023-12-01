#Pokemon Showdown Voice Controller
#Integrated Dialogue Manager: Bert Binary Classifier
#Aditya Pandey, Nitish Mallick, Savya Sachi Pandey, Vivek Kumar
#Note: Check the README and report for a full understanding of how this works

from poke_env import PlayerConfiguration
from poke_env.player import Player, RandomPlayer
from difflib import SequenceMatcher
import asyncio
import time
import speech_recognition as sr
import keyboard
from fuzzywuzzy import fuzz
import fuzzy
from transformers import BertModel
import torch
import torch.nn as nn
from transformers import BertTokenizer
import numpy as np
import pandas as pd
import openai
import time
import os
from functools import partial
import random
from collections import Counter
import tqdm
import re
import pyautogui


class BertBinaryClassifier(nn.Module):

    def __init__(self, d_hidden = 768, bert_variant = "bert-base-uncased"):


        super(BertBinaryClassifier, self).__init__()
        self.bert_layer = BertModel.from_pretrained(bert_variant)


        self.mlp_layer = torch.nn.Sequential(
            torch.nn.Linear(d_hidden,d_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(d_hidden,1)
        )

        self.sigmoid =torch.nn.Sigmoid()


    def forward(self, input, attn_mask):

        bert_output=self.bert_layer(input_ids=input, attention_mask=attn_mask).pooler_output
        mlp_output=self.mlp_layer(bert_output)
        final_output=self.sigmoid(mlp_output)

        return final_output