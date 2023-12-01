#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence speech recognition
with 🤗 Datasets' streaming mode.
"""
# You can also adapt this script for your own sequence to sequence speech
# recognition task. Pointers for this are left as comments.

import logging
import os
import sys
import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Iterable

import datasets
import torch
from datasets import DatasetDict, IterableDatasetDict, interleave_datasets, load_dataset
from torch.utils.data import IterableDataset

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from custom_trainer import Seq2SeqTrainerCustomLinearScheduler
from belarusian_text_normalizer import BelarusianTextNormalizer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.0.dev0")

require_version("datasets>=1.18.2", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class CustomTrainingArguments:
    """ Custom trianing arguments """
    
    learning_rate_end: Optional[float] = field(
        default=None,
        metadata={
            "help": ('Learning rate in the end of a training run. Passed to a Seq2SeqTrainerCustomLinearScheduler.')
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "feature extractor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    freeze_encoder: bool = field(
        default=False, metadata={"help": "Whether to freeze the entire encoder of the seq2seq model."}
    )
    forced_decoder_ids: List[List[int]] = field(
        default=None,
        metadata={
            "help": (
                "A list of pairs of integers which indicates a mapping from generation indices to token indices "
                "that will be forced before sampling. For example, [[0, 123]] means the first generated token "
                "will always be a token of index 123."
            )
        },
    )
    suppress_tokens: List[int] = field(
        default=None, metadata={"help": "A list of tokens that will be suppressed at generation."}
    )
    model_index_name: str = field(default=None, metadata={"help": "Pretty name for the model card."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    do_lower_case: bool = field(
        default=False,
        metadata={"help": "Whether the target text should be lower cased."},
    )
    do_remove_punctuation: bool = field(
        default=False,
        metadata={"help": "Whether the target text should be striped of punctuation."},
    )
    do_normalize_eval: bool = field(
        default=True,
        metadata={"help": "Whether to normalise the references and predictions in the eval WER calculation."},
    )
    language: str = field(
        default=None,
        metadata={
            "help": (
                "Language for multilingual fine-tuning. This argument should be set for multilingual fine-tuning "
                "only. For English speech recognition, it should be set to `None`."
            )
        },
    )
    task: str = field(
        default="transcribe",
        metadata={"help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."},
    )
    shuffle_buffer_size: Optional[int] = field(
        default=500,
        metadata={
            "help": (
                "The number of streamed examples to download before shuffling them. The large the buffer, "
                "the closer it is to real offline shuffling."
            )
        },
    )
    streaming_train: bool = field(
        default=True,
        metadata={"help": "Whether to use streaming mode to load and pre-process the train split."},
    )
    streaming_eval: bool = field(
        default=True,
        metadata={"help": "Whether to use streaming mode to load and pre-process the evaluation split."},
    )



@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def load_maybe_streaming_dataset(dataset_name, dataset_config_name, split="train", streaming=True, **kwargs):
    """
    Utility function to load a dataset in streaming mode. For datasets with multiple splits,
    each split is loaded individually and then splits combined by taking alternating examples from
    each (interleaving).
    """
    if "+" in split:
        # load multiple splits separated by the `+` symbol with streaming mode
        dataset_splits = [
            load_dataset(dataset_name, dataset_config_name, split=split_name, streaming=streaming, **kwargs)
            for split_name in split.split("+")
        ]
        # interleave multiple splits to form one dataset
        interleaved_dataset = interleave_datasets(dataset_splits)
        return interleaved_dataset
    else:
        # load a single split *with* streaming mode
        dataset = load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, **kwargs)
        return dataset


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((
        ModelArguments, DataTrainingArguments, 
        Seq2SeqTrainingArguments, CustomTrainingArguments
    ))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, custom_training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, custom_training_args = parser.parse_args_into_dataclasses()


    # 2. Setup logging
    now_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # update training_args if needed
    if custom_training_args.learning_rate_end is not None:
        logger.info(f'found learning_rate_end={custom_training_args.learning_rate_end} in passed arguments. '
                    'will pass it to training_args')
        training_args.learning_rate_end = custom_training_args.learning_rate_end
    else:
        logger.info(f'learning_rate_end is None. will not pass it to training_args')

    # log arguments
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data parameters: {data_args}")
    logger.info(f"Model parameters: {model_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()


    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        logger.info(f'output_dir already exists. will try to load last checkpoint.')
        
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            if training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
            else:
                logger.info(f'Last checkpoint found at: {last_checkpoint}. Will ignore it and resume training '
                            f'from passed resume_from_checkpoint param: {training_args.resume_from_checkpoint}')
                assert os.path.isdir(training_args.resume_from_checkpoint)
        else:    
            logger.info('last_checkpoint is None. will try to read from training_args.resume_from_checkpoint')
            
            if training_args.resume_from_checkpoint is not None and os.path.isdir(training_args.resume_from_checkpoint):
                logger.info(f'Will resume training from  passed resume_from_checkpoint param: '
                            f'{training_args.resume_from_checkpoint}')
            else:
                logger.info('last_checkpoint is None. resume_from_checkpoint is either None or not existing dir. '
                            'will try to read from the model saved in the root of output_dir.')

                dir_content = os.listdir(training_args.output_dir)
                if len(dir_content) == 0:
                    logger.info('output_dir is empty. will start training from scratch.')                    
                else:
                    model_fn = 'pytorch_model.bin'
                    if model_fn in dir_content:
                        logger.info(f'found {model_fn} inside output_dir. '
                                    f'will continue training treating output_dir as a last checkpoint.')
                        last_checkpoint = training_args.output_dir
                    else:
                        allowed_dirs = ['.git', '.gitattributes', 'src']
                        unexpected_content = set(dir_content).difference(allowed_dirs)
                        unexpected_content = [x for x in unexpected_content 
                                              if not x.endswith('.log') and os.path.isfile(x)]
                        if len(unexpected_content) > 0:
                            raise ValueError(
                                f'Could not find last_checkpoint, resume_from_checkpoint is either None '
                                'or not existing dir, output_dir is non-empty but does not contain a model.'
                                'Use --overwrite_output_dir to overcome. '
                                f'unexpected_content: {unexpected_content}'
                            )
                        else:
                            logger.info(f'dir is not empty, but contains only: {dir_content}. '
                                        'it is OK - will start training')
                        

    # Set seed before initializing model.
    set_seed(training_args.seed)


    # 4. Load dataset

    # TODO: replace dataset dicts with single key to IterableDataset and to Dataset.
    # don't know how to do it know - using dict simply because they work.
    raw_train = IterableDatasetDict() if data_args.streaming_train else DatasetDict()
    raw_eval = IterableDatasetDict() if data_args.streaming_eval else DatasetDict()

    if training_args.do_train:
        raw_train['train'] = load_maybe_streaming_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.train_split_name,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming_train,
        )

    if training_args.do_eval:
        raw_eval['eval'] = load_maybe_streaming_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.eval_split_name,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming_eval,
        )

    raw_datasets_features = list(next(iter(raw_train.values())).features.keys())

    if data_args.audio_column_name not in raw_datasets_features:
        raise ValueError(
            f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--audio_column_name` to the correct audio column - one of "
            f"{', '.join(raw_datasets_features)}."
        )

    if data_args.text_column_name not in raw_datasets_features:
        raise ValueError(
            f"--text_column_name {data_args.text_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--text_column_name` to the correct text column - one of "
            f"{', '.join(raw_datasets_features)}."
        )


    # 5. Load pretrained model, tokenizer, and feature extractor
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    config.update({"forced_decoder_ids": model_args.forced_decoder_ids, "suppress_tokens": model_args.suppress_tokens})

    if training_args.gradient_checkpointing:
        config.update({"use_cache": False})

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    if model_args.freeze_encoder:
        model.freeze_encoder()

    if data_args.language is not None:
        # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
        tokenizer.set_prefix_tokens(language=data_args.language, task=data_args.task)


    # 6. Explicitly resample speech dataset
    raw_train = raw_train.cast_column(
        data_args.audio_column_name, datasets.features.Audio(
            sampling_rate=feature_extractor.sampling_rate,
            mono=True
        )
    )
    raw_eval = raw_eval.cast_column(
        data_args.audio_column_name, datasets.features.Audio(
            sampling_rate=feature_extractor.sampling_rate,
            mono=True
        )
    )


    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    max_labels_length = 448  # model.config.max_length

    audio_column_name = data_args.audio_column_name
    text_column_name = data_args.text_column_name
    model_input_name = feature_extractor.model_input_names[0]
    do_lower_case = data_args.do_lower_case
    do_remove_punctuation = data_args.do_remove_punctuation
    normalizer = BelarusianTextNormalizer()  # custom normalizer based on 'official' text normalizer from OpenAI

    if data_args.max_train_samples is not None:
        raw_train['train'] = (
            raw_train['train'].take(data_args.max_train_samples)
            if data_args.streaming_train
            else raw_train['train'].select(range(data_args.max_train_samples))
        )

    if data_args.max_eval_samples is not None:
        raw_eval['eval'] = (
            raw_eval['eval'].take(data_args.max_eval_samples)
            if data_args.streaming_eval
            else raw_eval['eval'].select(range(data_args.max_eval_samples))
        )

    def prepare_dataset(sample, labels_max_len: int = None):
        # process audio
        audio = sample[audio_column_name]
        inputs = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"])
        # process audio length
        sample[model_input_name] = inputs.get(model_input_name)[0]
        sample["input_length"] = len(audio["array"])

        # process targets
        input_str = sample[text_column_name].lower() if do_lower_case else sample[text_column_name]
        if do_remove_punctuation:
            input_str = normalizer(input_str).strip()
        sample['labels'] = tokenizer(input_str).input_ids
        sample['labels_length'] = len(sample['labels'])  # include special characters

        sample['labels_truncated'] = 0
        # need to truncate validation and test labels that are longer that model.config.max_length.
        # can't drop such examples because this will affect validation and test scores.
        # thus need to truncate.
        if labels_max_len is not None:
            if len(sample['labels']) > labels_max_len:
                sample['labels'] = sample['labels'][:labels_max_len]
                sample['labels_truncated'] = 1

        return sample

    with training_args.main_process_first(desc="dataset map pre-processing"):
        logger.info(f'vectorizing dataset')

        # TODO: replace dataset dicts with single key to IterableDataset and to Dataset.
        # don't know how to do it know - using dict simply because they work.
        vectorized_train = IterableDatasetDict() if data_args.streaming_train else DatasetDict()
        vectorized_eval = IterableDatasetDict() if data_args.streaming_eval else DatasetDict()

        num_proc = None
        if data_args.streaming_train or data_args.streaming_eval:
            logger.info(f'will preprocess data using {num_proc} processes.')

        if data_args.streaming_train:
            vectorized_train['train'] = raw_train['train'].map(
                prepare_dataset, remove_columns=raw_datasets_features,
                fn_kwargs=dict(labels_max_len=None),
            ).with_format("torch")
        else:
            vectorized_train['train'] = raw_train['train'].map(
                prepare_dataset, remove_columns=raw_datasets_features,
                num_proc=num_proc,
                fn_kwargs=dict(labels_max_len=None),
            ).with_format("torch")

        if data_args.streaming_eval:
            vectorized_eval['eval'] = raw_eval['eval'].map(
                prepare_dataset, remove_columns=raw_datasets_features,
                fn_kwargs=dict(labels_max_len=max_labels_length),
            ).with_format("torch")
        else:
            vectorized_eval['eval'] = raw_eval['eval'].map(
                prepare_dataset, remove_columns=raw_datasets_features,
                num_proc=num_proc,
                fn_kwargs=dict(labels_max_len=max_labels_length),
            ).with_format("torch")

        if training_args.do_train and data_args.streaming_train:
            # manually shuffle if streaming (done by the trainer for non-streaming)
            vectorized_train['train'] = vectorized_train['train'].shuffle(
                buffer_size=data_args.shuffle_buffer_size,
                seed=training_args.seed,
            )

    # Filter training data that is shorter than min_input_length or longer than max_input_length.
    # Drop items with labels longer that max model length.
    # Drop such items from the train set only. Should keep them in eval set not to affect eval metrics.
    def is_audio_in_length_range(length):
        return min_input_length < length < max_input_length

    def are_labels_in_length_range(labels_length):
        return labels_length <= max_labels_length

    if training_args.do_train:
        # Filter items from train set only. 
        # Should keep them in eval set not to affect eval metrics.
        vectorized_train['train'] = vectorized_train['train'].filter(
            is_audio_in_length_range,
            input_columns=["input_length"],
        )
        vectorized_train['train'] = vectorized_train['train'].filter(
            are_labels_in_length_range,
            input_columns=["labels_length"],
        )


    # 8. Load Metric
    metric = evaluate.load("wer")
    do_normalize_eval = data_args.do_normalize_eval

    def compute_metrics(pred):
        pred_ids = pred.predictions

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

        if do_normalize_eval:
            pred_str = [normalizer(pred) for pred in pred_str]
            label_str = [normalizer(label) for label in label_str]
            # filtering step to only evaluate the samples that correspond to non-zero references:
            pred_str = [pred_str[i] for i in range(len(pred_str)) if len(label_str[i]) > 0]
            label_str = [label_str[i] for i in range(len(label_str)) if len(label_str[i]) > 0]

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}


    # 9. Create a single speech processor
    if is_main_process(training_args.local_rank):
        # save feature extractor, tokenizer and config
        feature_extractor.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)

    processor = AutoProcessor.from_pretrained(training_args.output_dir)


    # 10. Define data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )


    # 11. Configure Trainer
    # Trainer callback to reinitialise and reshuffle the streamable datasets at the beginning of each epoch
    # Only required for streaming: Trainer automatically shuffles non-streaming datasets
    class ShuffleCallback(TrainerCallback):
        def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
            if isinstance(train_dataloader.dataset, IterableDatasetShard):
                pass  # set_epoch() is handled by the Trainer
            elif isinstance(train_dataloader.dataset, IterableDataset):
                logger.info(f'ShuffleCallback. shuffling train dataset. '
                            f'seed: {training_args.seed}. dataset epoch: {train_dataloader.dataset._epoch}')
                train_dataloader.dataset.set_epoch(train_dataloader.dataset._epoch + 1)

    # Initialize Trainer
    trainer = Seq2SeqTrainerCustomLinearScheduler(
        model=model,
        args=training_args,
        train_dataset=vectorized_train['train'] if training_args.do_train else None,
        eval_dataset=vectorized_eval['eval'] if training_args.do_eval else None,
        tokenizer=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[ShuffleCallback()] if data_args.streaming_train else None,
    )


    # 12. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        logger.info(f'will launch training and pass resume_from_checkpoint={checkpoint}')
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the feature extractor too for easy upload

        metrics = train_result.metrics
        if data_args.max_train_samples:
            metrics["train_samples"] = data_args.max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


    # 13. Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )
        if data_args.max_eval_samples:
            metrics["eval_samples"] = data_args.max_eval_samples

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


    # 14. Write Training Stats
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "automatic-speech-recognition",
        "tags": "whisper-event",
    }
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name
        if "common_voice" in data_args.dataset_name:
            kwargs["language"] = data_args.dataset_config_name[:2]
        if model_args.model_index_name is not None:
            kwargs["model_name"] = model_args.model_index_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


if __name__ == "__main__":
    main()
