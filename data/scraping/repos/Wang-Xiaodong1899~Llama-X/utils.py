# -*- coding:utf-8 -*-

import openai
from openai import openai_object
import os
import sys
import shutil
import subprocess
import logging
import colorlog
import argparse
import copy
import pathlib
import shlex
import deepdish
from tqdm import tqdm
import time
import platform
import pickle
import yaml
import glob
import random
import msgpack
import importlib
import traceback
from PIL import Image
import functools
from functools import partial
import urllib.request
from warnings import simplefilter
from datetime import timedelta
from timeit import default_timer
from configobj import ConfigObj
import requests
import psutil
import hashlib
import imageio
import math
import h5py
import csv
import collections
from collections import OrderedDict
import json
import json_lines
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
from einops import rearrange, repeat
import torch.distributed as dist
from torchvision import datasets, transforms, utils
import torchvision
import cv2
from decord import VideoReader

import dataclasses
import io
from typing import Optional, Sequence, Union

StrOrOpenAIObject = Union[str, openai_object.OpenAIObject]
image_placeholder = "[IMG]" + "<image>" * 32 + "[/IMG]"

# Disable transformers outputs weights.
logging.getLogger().setLevel(logging.WARNING)
simplefilter(action='ignore', category=FutureWarning)


def get_logger(filename=None):
    """
    examples:
        logger = get_logger('try_logging.txt')

        logger.debug("Do something.")
        logger.info("Start print log.")
        logger.warning("Something maybe fail.")
        try:
            raise ValueError()
        except ValueError:
            logger.error("Error", exc_info=True)

        tips:
        DO NOT logger.inf(some big tensors since color may not helpful.)
    """
    logger = logging.getLogger('utils')
    level = logging.DEBUG
    logger.setLevel(level=level)
    # Use propagate to avoid multiple loggings.
    logger.propagate = False
    # Remove %(levelname)s since we have colorlog to represent levelname.
    format_str = '[%(asctime)s <%(filename)s:%(lineno)d> %(funcName)s] %(message)s'

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(level)
    coloredFormatter = colorlog.ColoredFormatter(
        '%(log_color)s' + format_str,
        datefmt='%Y-%m-%d %H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            # 'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'reg,bg_white',
        }
    )

    streamHandler.setFormatter(coloredFormatter)
    logger.addHandler(streamHandler)

    if filename:
        fileHandler = logging.FileHandler(filename)
        fileHandler.setLevel(level)
        formatter = logging.Formatter(format_str)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    # Fix multiple logging for torch.distributed
    try:
        class UniqueLogger:
            def __init__(self, logger):
                self.logger = logger
                self.local_rank = torch.distributed.get_rank()

            def info(self, msg, *args, **kwargs):
                if self.local_rank == 0:
                    return self.logger.info(msg, *args, **kwargs)

            def warning(self, msg, *args, **kwargs):
                if self.local_rank == 0:
                    return self.logger.warning(msg, *args, **kwargs)

        logger = UniqueLogger(logger)
    # AssertionError for gpu with no distributed
    # AttributeError for no gpu.
    except Exception:
        pass
    return logger


logger = get_logger()
logger.info("<utils.py>: Deep Learning Utils @ Chenfei Wu")



@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    suffix: Optional[str] = None
    logprobs: Optional[int] = None
    echo: bool = False


def openai_completion(
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
    model_name="text-davinci-003",
    sleep_time=2,
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    **decoding_kwargs,
) -> Union[Union[StrOrOpenAIObject], Sequence[StrOrOpenAIObject], Sequence[Sequence[StrOrOpenAIObject]],]:
    """Decode with OpenAI API.

    Args:
        prompts: A string or a list of strings to complete. If it is a chat model the strings should be formatted
            as explained here: https://github.com/openai/openai-python/blob/main/chatml.md. If it is a chat model
            it can also be a dictionary (or list thereof) as explained here:
            https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        batch_size: Number of prompts to send in a single request. Only for non chat model.
        max_instances: Maximum number of prompts to decode.
        max_batches: Maximum number of batches to decode. This argument will be deprecated in the future.
        return_text: If True, return text instead of full completion object (which contains things like logprob).
        decoding_kwargs: Additional decoding arguments. Pass in `best_of` and `logit_bias` if you need them.

    Returns:
        A completion or a list of completions.
        Depending on return_text, return_openai_object, and decoding_args.n, the completion type can be one of
            - a string (if return_text is True)
            - an openai_object.OpenAIObject object (if return_text is False)
            - a list of objects of the above types (if decoding_args.n > 1)
    """
    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    completions = []
    for batch_id, prompt_batch in tqdm.tqdm(
        enumerate(prompt_batches),
        desc="prompt_batches",
        total=len(prompt_batches),
    ):
        batch_decoding_args = copy.deepcopy(decoding_args)  # cloning the decoding_args

        while True:
            try:
                shared_kwargs = dict(
                    model=model_name,
                    **batch_decoding_args.__dict__,
                    **decoding_kwargs,
                )
                completion_batch = openai.Completion.create(prompt=prompt_batch, **shared_kwargs)
                choices = completion_batch.choices

                for choice in choices:
                    choice["total_tokens"] = completion_batch.usage.total_tokens
                completions.extend(choices)
                break
            except openai.error.OpenAIError as e:
                logging.warning(f"OpenAIError: {e}.")
                if "Please reduce your prompt" in str(e):
                    batch_decoding_args.max_tokens = int(batch_decoding_args.max_tokens * 0.8)
                    logging.warning(f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying...")
                else:
                    logging.warning("Hit request rate limit; retrying...")
                    time.sleep(sleep_time)  # Annoying rate limit on requests.

    if return_text:
        completions = [completion.text for completion in completions]
    if decoding_args.n > 1:
        # make completions a nested list, where each entry is a consecutive decoding_args.n of original entries.
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        # Return non-tuple if only 1 input and 1 generation.
        (completions,) = completions
    return completions


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def split_filename(filename):
    absname = os.path.abspath(filename)
    dirname, basename = os.path.split(absname)
    split_tmp = basename.rsplit('.', maxsplit=1)
    if len(split_tmp) == 2:
        rootname, extname = split_tmp
    elif len(split_tmp) == 1:
        rootname = split_tmp[0]
        extname = None
    else:
        raise ValueError("programming error!")
    return dirname, rootname, extname


def file2data(filename, type=None, printable=True, **kwargs):
    dirname, rootname, extname = split_filename(filename)
    print_load_flag = True
    if type:
        extname = type
    if extname == 'pkl':
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    elif extname == 'msg':
        with open(filename, 'rb') as f:
            data = msgpack.load(f, encoding="utf-8")
    elif extname == 'h5':
        split_num = kwargs.get('split_num')
        if split_num:
            print_load_flag = False
            if isinstance(split_num, int):
                filenames = ["%s_%i" % (filename, i) for i in range(split_num)]
                if split_num != len(glob.glob("%s*" % filename)):
                    print('Maybe you are giving a wrong split_num(%d) != seached num (%d)' % (
                        split_num, len(glob.glob("%s*" % filename))))

            elif split_num == 'auto':
                filenames = glob.glob("%s*" % filename)
                print('Auto located %d splits linked to %s' % (len(filenames), filename))
            else:
                raise ValueError("params['split_num'] got unexpected value: %s, which is not supported." % split_num)
            data = []
            for e in filenames:
                data.extend(deepdish.io.load(e))
            print('Loaded data from %s_(%s)' % (
                os.path.abspath(filename), ','.join(sorted([e.split('_')[-1] for e in filenames]))))
        else:
            data = deepdish.io.load(filename)
    elif extname == 'csv':
        data = pd.read_csv(filename)
    elif extname == 'tsv':  # Returns generator since tsv file is large.
        if not kwargs.get('delimiter'):  # Set default delimiter
            kwargs['delimiter'] = '\t'
        if not kwargs.get('fieldnames'):  # Check field names
            raise ValueError('You must specify fieldnames when load tsv data.')
        # Required args.
        key_str = kwargs.pop('key_str')
        decode_fn = kwargs.pop('decode_fn')
        # Optimal args.
        topk = kwargs.pop('topk', None)
        redis = kwargs.pop('redis', None)
        if not redis:
            data = dict()
        else:
            data = redis
        if not redis or not redis.check():
            with open(filename) as f:
                reader = csv.DictReader(f, **kwargs)
                for i, item in enumerate(tqdm(reader)):
                    if not redis:  # if memory way
                        decode_fn(item)
                    data[item[key_str]] = item
                    if topk is not None and i + 1 == topk:
                        break
        else:
            print('check_str %s in redis, skip loading.' % data.check_str)
    elif extname == 'hy':
        data = h5py.File(filename, 'r')
    elif extname in ['npy', 'npz']:
        try:
            data = np.load(filename, allow_pickle=True)
        except UnicodeError:
            print('%s is python2 format, auto use latin1 encoding.' % os.path.abspath(filename))
            data = np.load(filename, encoding='latin1', allow_pickle=True)
    elif extname == 'json':
        with open(filename) as f:
            try:
                data = json.load(f)
            except json.decoder.JSONDecodeError as e:
                raise ValueError('[error] utils.file2data: failed to load json file %s' % filename)
    elif extname == 'jsonl':
        with open(filename, 'rb') as f:
            data = [e for e in json_lines.reader(f)]
    elif extname == 'ini':
        data = ConfigObj(filename, encoding='utf-8')
    elif extname in ['pth', 'ckpt']:
        data = torch.load(filename, map_location=kwargs.get('map_location'))
    elif extname == 'txt':
        top = kwargs.get('top', None)
        with open(filename, encoding='utf-8') as f:
            if top:
                data = [f.readline() for _ in range(top)]
            else:
                data = [e for e in f.read().split('\n') if e]
    elif extname == 'yaml':
        with open(filename, 'r') as f:
            data = yaml.load(f)
    else:
        raise ValueError('type can only support h5, npy, json, txt')
    if printable:
        if print_load_flag:
            print('Loaded data from %s' % os.path.abspath(filename))
    return data

def adaptively_load_state_dict(target, state_dict):
    target_dict = target.state_dict()

    try:
        common_dict = {k: v for k, v in state_dict.items() if k in target_dict and v.size() == target_dict[k].size()}
    except Exception as e:
        print('load error %s', e)
        common_dict = {k: v for k, v in state_dict.items() if k in target_dict}

    if 'param_groups' in common_dict and common_dict['param_groups'][0]['params'] != \
            target.state_dict()['param_groups'][0]['params']:
        print('Detected mismatch params, auto adapte state_dict to current')
        common_dict['param_groups'][0]['params'] = target.state_dict()['param_groups'][0]['params']
    target_dict.update(common_dict)
    target.load_state_dict(target_dict)

    missing_keys = [k for k in target_dict.keys() if k not in common_dict]
    unexpected_keys = [k for k in state_dict.keys() if k not in common_dict]

    if len(unexpected_keys) != 0:
        print(
            f"Some weights of state_dict were not used in target: {unexpected_keys}"
        )
    if len(missing_keys) != 0:
        print(
            f"Some weights of state_dict are missing used in target {missing_keys}"
        )
    if len(unexpected_keys) == 0 and len(missing_keys) == 0:
        print("Strictly Loaded state_dict.")

def get_index(num_frames, num_segments):
    print(f"===> num_frames: {num_frames}, num_segments: {num_segments}")
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def process_img(img_path=None, img=None, device=torch.device("cuda")):
    assert img_path is not None or img is not None, "you should pass either path to an image or a PIL image object"
    width, height = 224, 224
    OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
    if img_path:
        img = Image.open(img_path).convert("RGB")
    img = img.resize((width, height))
    img = np.array(img) / 255.
    img = (img - OPENAI_DATASET_MEAN) / OPENAI_DATASET_STD
    img = torch.tensor(img).to(device).to(torch.float)
    img = torch.einsum('hwc->chw', img)
    img = img.unsqueeze(0)
    return img


def process_video(video_path=None):
    vr = VideoReader(video_path)
    frame_indices = get_index(len(vr), 8)
    image_list = []
    text_sequence = ''
    for frame_index in frame_indices:
        image = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        image = process_img(img=image)
        image_list.append(image)
        text_sequence += image_placeholder
    return image_list, text_sequence


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class ParameterList(object):
    """a mock parameter list object until below issue is resolved
        https://github.com/pytorch/pytorch/issues/36035
    """

    def __init__(self, kls, prefix, length):
        self.ind = 0
        self.kls = kls
        self.prefix = prefix
        self.length = length

    def _keyname(self, prefix, ind):
        return f'{prefix}_{ind}'

    def append(self, x):
        setattr(self.kls, self._keyname(self.prefix, self.ind), x)
        self.ind += 1

    def to_list(self):
        return [getattr(self.kls, self._keyname(self.prefix, i)) for i in range(self.length)]

from functools import partial, reduce

class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, axial_shape, axial_dims=None):
        super().__init__()

        self.dim = dim
        axial_shape = tuple([shape for shape in axial_shape if shape != 1])  # remove the axis whose shape is 1
        self.shape = axial_shape
        self.max_seq_len = reduce(lambda x, y: x * y, axial_shape, 1)

        self.summed = axial_dims is None
        axial_dims = ((dim,) * len(axial_shape)) if self.summed else axial_dims

        assert len(self.shape) == len(
            axial_dims), 'number of axial dimensions must equal the number of dimensions in the shape'
        assert self.summed or not self.summed and sum(
            axial_dims) == dim, f'axial dimensions must sum up to the target dimension {dim}'

        self.weights = ParameterList(self, 'weights', len(axial_shape))

        for ind, (shape, axial_dim) in enumerate(zip(self.shape, axial_dims)):
            ax_shape = [1] * len(self.shape)
            ax_shape[ind] = shape
            ax_shape = (1, *ax_shape, axial_dim)
            ax_emb = nn.Parameter(torch.zeros(ax_shape).normal_(0, 1))
            self.weights.append(ax_emb)

    def forward(self, x, idx=None, row_range=None, col_range=None):
        b, t, e = x.shape
        assert (t <= self.max_seq_len), f'Sequence length ({t}) must ' \
                                        f'be less than the maximum sequence length allowed ({self.max_seq_len})'
        embs = []

        for ax_emb in self.weights.to_list():
            axial_dim = ax_emb.shape[-1]
            expand_shape = (b, *self.shape, axial_dim)
            emb = ax_emb.expand(expand_shape).reshape(b, self.max_seq_len, axial_dim)
            embs.append(emb)
        pos_emb = sum(embs) if self.summed else torch.cat(embs, dim=-1)

        if idx is None and row_range is None:
            return pos_emb[:, :t].to(x)
        elif idx is not None:
            return pos_emb[:, idx:idx + 1].to(x)
        else:
            rows = torch.arange(row_range[0], row_range[1])
            cols = torch.arange(col_range[0], col_range[1])
            coords = torch.stack(torch.meshgrid([rows, cols]), dim=-1).reshape(-1, 2)
            coords[:, 0] = coords[:, 0] * self.shape[0]
            pos_idxs = coords.sum(dim=-1)
            return pos_emb[:, pos_idxs].to(x)


def data2file(data, filename, type=None, override=False, printable=False, **kwargs):
    dirname, rootname, extname = split_filename(filename)
    print_did_not_save_flag = True
    if type:
        extname = type
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    if not os.path.exists(filename) or override:
        if extname == 'pkl':
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
        elif extname == 'msg':
            with open(filename, 'wb') as f:
                msgpack.dump(data, f)
        elif extname == 'csv':
            with open(filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(data)
        elif extname == 'json':
            with open(filename, 'w') as f:
                json.dump(data, f)
        elif extname == 'npy':
            np.save(filename, data)
        elif extname in ['jpg', 'png', 'jpeg']:
            utils.save_image(data, filename, **kwargs)
        elif extname == 'gif':
            if not kwargs.get('fps'):
                raise DeprecationWarning("duration is deprecated, please specify fps directly.")
            imageio.mimsave(filename, data, format='GIF', fps=kwargs.get('fps'))
        # elif extname == 'ckpt':
        #     tf.train.Saver().save(data, filename)
        # elif extname == 'jpg' or extname == 'png':
        #     plt.imsave(filename, data)
        elif extname == 'pth':
            torch.save(data, filename)
        elif extname == 'txt':
            if kwargs is None:
                kwargs = {}
            max_step = kwargs.get('max_step')
            if max_step is None:
                max_step = np.Infinity

            with open(filename, 'w', encoding='utf-8') as f:
                for i, e in enumerate(data):
                    if i < max_step:
                        f.write(str(e) + '\n')
                    else:
                        break
        else:
            raise ValueError('type can only support h5, csv, json, sess')
        if printable: logger.info('Saved data to %s' % os.path.abspath(filename))
    else:
        if print_did_not_save_flag: logger.info(
            'Did not save data to %s because file exists and override is False' % os.path.abspath(
                filename))
