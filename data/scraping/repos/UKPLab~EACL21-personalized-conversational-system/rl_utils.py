from pytorch_pretrained_bert import OpenAIAdam, OpenAIGPTTokenizer, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME
from modeling_openai import OpenAIGPTDoubleHeadsModel
from modeling_gpt2 import GPT2DoubleHeadsModel
import re
from collections import Counter
import numpy as np
import nltk
nltk.data.path.append("./nltk_data/")
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from utils import SPECIAL_TOKENS
from persona_consistency_subreward.nli_task import main as nli_engine

import itertools

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import file_utils
from more_itertools import chunked
import torch
import string
#from pattern.text.en import lemma
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
lemma = wordnet_lemmatizer.lemmatize
from stop_words import get_stop_words
import os
import sys
import pickle as pk
import random

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
PUNCTUATION = str.maketrans('', '', string.punctuation)
STOP_WORDS = None


def create_critic(args, rl=False):
    """
    Not applied.
    :param args:
    :param rl:
    :return:
    """
    critic, critic_optim = create_model(args, rl)
    if args.device == 'cuda':
        critic.cuda()
    return critic, critic_optim


def create_optim(model, args, rl=False):
    """
    Not applied.
    :param model:
    :param args:
    :param rl:
    :return:
    """
    if not rl:
        optimizer = OpenAIAdam(model.parameters(), lr=args.lr)
    else:
        optimizer = OpenAIAdam(model.parameters(), lr=args.reinforce_lr)
    return optimizer


def create_model(args, rl=False):
    """
    Not applied.
    :param args:
    :param rl:
    :return:
    """
    model_class = GPT2DoubleHeadsModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
    model = model_class.from_pretrained(args.model_checkpoint)
    optim = create_optim(model, args, rl)
    return model, optim


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace.(Source: ParlAI metric.py line 56)"""
    s = s.lower()

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_rewarder(pred, label):
    """Return the F1 score between the guess and *any* answer."""
    if pred is None or label is None:
        return 0
    pred_tokens = normalize_answer(pred).split()
    label_tokens = normalize_answer(label).split()

    common = Counter(pred_tokens) & Counter(label_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(label_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def bleu_rewarder(pred, label):
    """Return the BLEU score between the guess and *any* answer."""
    if pred is None or label is None:
        return 0
    pred_tokens = normalize_answer(pred).split()
    label_tokens = normalize_answer(label).split()

    bleu_2 = bleu_score.sentence_bleu([label_tokens], pred_tokens, smoothing_function=SmoothingFunction().method1, weights=(0.5, 0.5, 0.0, 0.0))
    return bleu_2


def plot_reward(metrics, metric_type, folder, name, interval):
    """
    Plot the reward variation tendency during training.
    :param metrics: all metrics.
    :param metric_type: the metric going to be plot, e.g. persona_reward.
    :param folder: target folder for saving the figure.
    :param name: target file name for saving the figure.
    :param interval: plot interval.
    :return: saved figure in the target folder.
    """
    l = metrics[metric_type]
    ave_interval = [sum(x) / len(x) for x in chunked(l, interval)]
    for i, x in enumerate(ave_interval):
        if x >= 100:   # limit the maximum value in the figure
            ave_interval[i] = 100
    y = np.array(ave_interval)
    x = np.arange(len(y))
    x = interval * x
    plt.figure()
    # plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    # plt.plot(x, y, "b--", linewidth=1)
    plt.xlabel('Batch')   # x axis label
    plt.ylabel('Value')   # y axis label
    plt.title('Evaluation Result of ' + metric_type)   # figure title
    file_utils.save_pic(plt, folder, name)
    # plt.show()
    # plt.savefig("line.jpg")
    print('INFO - rl_utils: rewards saved!')


class LinearRegressionModel(torch.nn.Module):
    """
    Linear layer model (critic model).
    """
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(40483, 1)  # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


def process_document(document):   # Tokenize
    """
    Pre-process a sentence.
    :param document: the sentence to be pre-processed.
    :return: pre-precessed sentence.
    """
    tokens = []
    for token in nltk.word_tokenize(document):
        token = token.translate(PUNCTUATION)   # Clean punctuation
        token = token.lower()   # Convert to lower case
        if token.lower() in STOP_WORDS:   # Ignore stopwords
            continue
        if not token.isalpha():   # Filter non-alphabetical tokens
            continue
        if len(token)<=2:   # Remove short words
            continue
         #token = PorterStemmer().stem(token)   # Stem
        tokens.append(lemma(token))
        # tokens.append(token)
    return tokens


def init_stop_words():
    with open('stopwords.txt', 'r') as handle:
        stopwords_customized = []
        for line in handle:
            if line == '\n':
                pass
            else:
                stopwords_customized.append(line.replace('\n', ''))
    stop_words = get_stop_words('en') + [',', '.', '!', '?']
    stop_words.remove('not')
    global STOP_WORDS
    STOP_WORDS = stop_words + stopwords_customized


def read_model(folder, name, type):
    """
    Read a pickled model from file.
    :param folder: folder for the model.
    :param name: model name.
    :param type: model type (suffix).
    :return: the loaded model.
    """
    base_dir = os.path.dirname(__file__)
    f = open(os.path.join(base_dir, folder, name + type), 'rb')
    try:
        model = pk.load(f)
    except:
        print("Unexpected SAVING error:", sys.exc_info())
        model = None
    finally:
        f.close()
        return model


def tokens_to_vector(tokenized_text, glove_model):
    """Convert tokens to vector (GloVe-based, not applied)."""
    vec = []
    if tokenized_text == []:
        vec = [[0 for _ in range(300)]]
    else:
        for token in tokenized_text:
            try:
                vec.append(glove_model.wv[token])
            except:
                continue
    # max_len = max(max_len, len(vec))
    if vec == []:
        # continue
        vec = [[0 for _ in range(300)]]
    vec = np.array(vec)
    return vec.mean(axis=0)


def bert_vector(text, tokenizer, model, args):
    """
    BERT-based embedding.
    :param text: original text to be embedded.
    :param tokenizer: BERT tokenizer.
    :param model: BERT model.
    :param args: args.
    :return: BERT embedded sentence.
    """
    text = '[CLS] ' + text + ' [SEP]'
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.LongTensor([indexed_tokens]).to(args.device)
    layer_output = model(tokens_tensor, output_all_encoded_layers=True)
    encoded_layer = np.array(np.mean(layer_output[0][-2].cpu().detach().numpy()[0], axis=0))   # take the second-to-last layer as ecoded layer.
    return encoded_layer


def reset_seed(seed):
    """
    Reset the seed.
    :param seed: seed number.
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
