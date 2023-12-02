import os
import shutil
import sys
import heapq
import bisect
import operator
import pickle
import json
import nltk
from nltk import sent_tokenize, word_tokenize
import math
import statistics as stat
import time
import datetime as dt
import logging
from logging import Logger, Handler
import random
import threading as td
import multiprocessing as mp
import re
from re import Match
import unittest as ut
import platform
import csv
from enum import Enum
from inspect import getmembers, isfunction, ismethod
from pprint import pprint
from dataclasses import dataclass, field
#
import gradio as gr
from warnings import filterwarnings
from operator import itemgetter, attrgetter
from pprint import pprint
from itertools import (
    chain, accumulate, product, permutations, combinations, combinations_with_replacement,
    compress, starmap, zip_longest

)
from functools import partial, cache, lru_cache, cmp_to_key, reduce
from copy import copy, deepcopy
from argparse import ArgumentParser, Namespace
from queue import Queue, SimpleQueue, PriorityQueue
from hashlib import sha256
from typing import (
    Literal, List, Tuple, Dict, Set, Callable, Optional, Union, Any,
    Deque, NamedTuple, DefaultDict, Counter, OrderedDict,
    Sequence, Mapping, Iterable, Iterator, TypeVar, Generic, Generator
)
from typing_extensions import TypeAlias, Self
from types import SimpleNamespace
# from collections import deque, namedtuple, OrderedDict, defaultdict, Counter  # use typing
from _collections_abc import dict_items, dict_keys, dict_values
from contextlib import contextmanager
from numbers import Number
from fractions import Fraction
import pyximport
#
from flask import Flask, url_for, render_template
import yaml
from sortedcontainers import SortedList, SortedDict, SortedSet
from tqdm import tqdm
import numpy as np
from numpy import ndarray
from numpy.random import RandomState
from numpy.typing import NDArray, ArrayLike
import pandas as pd
from pandas import DataFrame, Series
#
import numba
from numba import jit, njit, vectorize, guvectorize
from numba.core.types import (
    void, uint8, int32, int64, float16, float32, float64, boolean, string,
    ListType, List as ReflectList, Array
)
from numba.typed.typedlist import List as TypedList
from numba.typed.typeddict import Dict as TypedDict
from numba import typeof
#
from urllib.parse import urljoin
from urllib.error import HTTPError
from urllib.request import urlretrieve
import requests
from lxml import etree
#
from xml.etree.ElementTree import ElementTree as ET, Element
from lxml.etree import _Element as Element2
from selenium.webdriver.remote.webelement import WebElement
#
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import Keys, Proxy, ActionChains
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.common.exceptions import NoSuchElementException
#
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
from PIL import Image
import cv2 as cv
#
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.manifold import TSNE
#
import torch
from torch import Tensor, dtype as Dtype, device as Device, Generator as TGenerator
from torch.nn import Module
import torch.linalg as tl
from torch.optim import Optimizer
from torch.nn.parameter import Parameter
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd.function import FunctionCtx, Function
from torch.optim import lr_scheduler as lrs
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
import torch.nn.init as init
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval
from torch.nn.parallel import DataParallel as DP, DistributedDataParallel as DDP
from torch.utils.data import (
    Dataset, IterableDataset, TensorDataset,
    Sampler, RandomSampler, SequentialSampler, BatchSampler, DistributedSampler,
    DataLoader, default_collate, get_worker_info,
    random_split
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
import torch.utils.data as tud
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn.modules.module import _IncompatibleKeys as IncompatibleKeys
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
# 
from peft.utils.config import PeftConfig
from peft.peft_model import PeftModelForCausalLM, PeftModel
from peft.tuners.lora import LoraConfig
#
import torchvision.transforms._functional_tensor as tvtF_t
import torchvision.transforms._functional_pil as tvtF_pil
import torchvision.transforms.functional as tvtF
from torchvision.transforms.functional import InterpolationMode, pil_modes_mapping
import torchvision as tv
import torchvision.transforms as tvt
import torchvision.datasets as tvd
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100, STL10
from torchvision.utils import make_grid, save_image
import torchvision.models as tvm
from torchvision.models import ResNet, DenseNet, resnet18
#
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import LightningCLI
from lightning_utilities.core.rank_zero import rank_zero_only
#
from transformers.pipelines import pipeline
from transformers.generation.streamers import TextStreamer
from transformers.models.auto.modeling_auto import (AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, 
                                                    AutoModelForSequenceClassification, AutoModelForQuestionAnswering, 
                                                    AutoModelForMultipleChoice, AutoModelForTokenClassification)
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.data.data_collator import DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from datasets.load import load_dataset
from datasets.combine import concatenate_datasets
#
from modelscope.msdatasets import MsDataset
from modelscope.hub.snapshot_download import snapshot_download
#
from torchmetrics import MeanMetric, Metric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.precision_recall import Precision, Recall
from torchmetrics.classification.f_beta import F1Score, FBetaScore
from torchmetrics.classification.auroc import AUROC
from torchmetrics.classification.average_precision import AveragePrecision
# 使用libs_ml中的metrics. (比torchmetrics.functional更快)
#
import gym
from gym import Env
import openai
from flask import Flask, url_for, render_template, redirect, request
from markupsafe import Markup
from wtforms import StringField, Form, SubmitField, PasswordField, BooleanField
from wtforms.validators import DataRequired, Length
#
import mini_lightning as ml
# _remove_keys, _key_add_suffix
