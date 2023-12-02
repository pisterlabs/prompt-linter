import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

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

from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from arguments import ModelArguments,DataTrainingArguments 
from collator import DataCollatorSpeechSeq2SeqWithPadding
import wandb
import tensorboard

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.0.dev0")
require_version("datasets>=1.18.2", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")

logger = logging.getLogger(__name__)

def load_maybe_streaming_dataset(dataset_name, dataset_config_name, split="train", streaming=False, **kwargs):
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
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a yaml file,
        model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    training_args.hub_token = os.getenv("HF_TOKEN","<HF_TOKEN>")
    print(training_args.hub_token)
    # 2. Setup logging
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
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 4. Load dataset
    raw_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()
    #raw_datasets = DatasetDict()

    if training_args.do_train:
        raw_datasets["train"] = load_maybe_streaming_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.train_split_name,
            use_auth_token=training_args.hub_token if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )

    if training_args.do_eval:
        raw_datasets["eval"] = load_maybe_streaming_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.eval_split_name,
            use_auth_token=training_args.hub_token if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )

    raw_datasets_features = list(next(iter(raw_datasets.values())).features.keys())

    #Validation of the data and columns to be use
    #Evaluate if the audio column is in the dataset
    if data_args.audio_column_name not in raw_datasets_features:
        raise ValueError(
            f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--audio_column_name` to the correct audio column - one of "
            f"{', '.join(raw_datasets_features)}."
        )

    #Evaluate if the text column is in the dataset
    if data_args.text_column_name not in raw_datasets_features:
        raise ValueError(
            f"--text_column_name {data_args.text_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--text_column_name` to the correct text column - one of "
            f"{', '.join(raw_datasets_features)}."
        )

    # 5. Load pretrained model, tokenizer, and feature extractor

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=training_args.hub_token if model_args.use_auth_token else None,
    )

    config.update({"forced_decoder_ids": model_args.forced_decoder_ids, "suppress_tokens": model_args.suppress_tokens})

    if training_args.gradient_checkpointing:
        config.update({"use_cache": False})


    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=training_args.hub_token if model_args.use_auth_token else None,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=training_args.hub_token if model_args.use_auth_token else None,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=training_args.hub_token if model_args.use_auth_token else None,
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    if model_args.freeze_encoder:
        model.freeze_encoder()
    
    # For Smaller datasets we can use Dropout max 0.1
    if model_args.model_dropout is not None:
        model.config.update({"dropout": model_args.model_dropout})

    if data_args.language is not None:
        # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
        tokenizer.set_prefix_tokens(language=data_args.language, task=data_args.task)

    # 6. Resample speech dataset if necessary
    #Resamplig if necessary
    dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.audio_column_name].sampling_rate

    if dataset_sampling_rate != feature_extractor.sampling_rate:
        raw_datasets = raw_datasets.cast_column(
            data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
        )

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    audio_column_name = data_args.audio_column_name
    text_column_name = data_args.text_column_name
    model_input_name = feature_extractor.model_input_names[0]
    do_lower_case = data_args.do_lower_case
    do_remove_punctuation = data_args.do_remove_punctuation
    
    normalizer = BasicTextNormalizer()  # 'official' text normalizer from OpenAI

    if data_args.max_train_samples is not None:
        raw_datasets["train"] = (
            raw_datasets["train"].take(data_args.max_train_samples)
            if data_args.streaming
            else raw_datasets["train"].select(range(data_args.max_train_samples))
        )

    if data_args.max_eval_samples is not None:
        raw_datasets["eval"] = (
            raw_datasets["eval"].take(data_args.max_eval_samples)
            if data_args.streaming
            else raw_datasets["eval"].select(range(data_args.max_eval_samples))
        )

    def prepare_dataset(batch):
        # process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]
        batch["input_length"] = len(sample["array"])

        # process targets
        input_str = batch[text_column_name].lower() if do_lower_case else batch[text_column_name]
        if do_remove_punctuation:
            input_str = normalizer(input_str).strip()
        batch["labels"] = tokenizer(input_str).input_ids
        return batch

    with training_args.main_process_first(desc="dataset map pre-processing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=raw_datasets_features,
        ).with_format("torch")

        if training_args.do_train and data_args.streaming:
            # manually shuffle if streaming (done by the trainer for non-streaming)
            vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(
                buffer_size=data_args.shuffle_buffer_size,
                seed=training_args.seed,
            )

    # filter training data that is shorter than min_input_length or longer than
    # max_input_length
    def is_audio_in_length_range(length):
        return min_input_length < length < max_input_length

    if training_args.do_train:
        vectorized_datasets["train"] = vectorized_datasets["train"].filter(
            is_audio_in_length_range,
            input_columns=["input_length"],
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
                train_dataloader.dataset.set_epoch(train_dataloader.dataset._epoch + 1)

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[ShuffleCallback()] if data_args.streaming else None,
    )


    # 11.1 Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("********************")
        logger.info("*** Evaluate *******")
        logger.info("********************")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )
        #if data_args.max_eval_samples:
        #    metrics["eval_samples"] = data_args.max_eval_samples
        #trainer.log_metrics("eval", metrics)
        #trainer.save_metrics("eval", metrics)
        logger.info(metrics)
        logger.info("********************")
        logger.info("*** Evaluate end ***")
        logger.info("********************")

    logger.info("# 12. Training #######################################################")
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the feature extractor too for easy upload

        metrics = train_result.metrics
        if data_args.max_train_samples:
            metrics["train_samples"] = data_args.max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    
    logger.info("# 13. Evaluation #######################################################")
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
        logger.info(metrics)
        logger.info("*** Evaluate end ***")
    
    logger.info("# 14. Write Training Stats #######################################################")
    
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "automatic-speech-recognition"
    }

    if data_args.dataset_name is not None:
        #kwargs["dataset_tags"] = data_args.dataset_name
        #kwargs["dataset"] = data_args.dataset_name
        kwargs["model_name"] = model_args.model_index_name

    if training_args.push_to_hub:
        
        trainer.push_to_hub(**kwargs)
        
        feature_extractor.save_pretrained(
            training_args.output_dir
            ,push_to_hub=training_args.push_to_hub
            ,repo_id=training_args.hub_model_id
            ,use_auth_token=training_args.hub_token if model_args.use_auth_token else None
        )

    else:
        trainer.create_model_card(**kwargs)

    return results


if __name__ == "__main__":
    main()
