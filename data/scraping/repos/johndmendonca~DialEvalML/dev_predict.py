from argparse import ArgumentParser, BooleanOptionalAction
from transformers import (
    Trainer,
    AutoConfig,
    AutoModelWithHeads,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
    XLMRobertaForMaskedLM,
)
from transformers.utils.logging import set_verbosity_info
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
import logging
import torch
import pandas as pd
from datetime import datetime
import json
from scipy import stats
from pathlib import Path
from sklearn.model_selection import train_test_split

from DialEvalML.utils.utils import *
from DialEvalML.mlm import MLM_predictor

import openai


openai.api_key = "key"
MAX_LENGTH = 256
DIAL_LVL_DATASETS = ["fed-dial", "persona-see"]

# Models
models = {
    "nsp": {
        "nsp_paL_siamese": "exp/xlm-roberta-large/nsp/paL_siamese3",
        "nsp_paL_concat": "exp/xlm-roberta-large/nsp/paL",
        "nsp_ml75": "exp/xlm-roberta-large/nsp/ml75",
    },
    "vsp": {
        "vsp_ml5": "exp/xlm-roberta-large/vsp/ml5",
        "vsp_en": "exp/xlm-roberta-large/vsp/en",
        "vsp_paL": "exp/xlm-roberta-large/vsp/paL",
    },
    "engagement": {
        "endex_ml10": "exp/xlm-roberta-large/eng/endex_ml10",
        "endex_ml20": "exp/xlm-roberta-large/eng/endex_ml20",
        "endex_ml50": "exp/xlm-roberta-large/eng/endex_ml50",
    },
}

# Dev files
dev = {
    "convai2-grade": {
        "path": "data/DSTC_11_Track_4/metadata/dev/convai2-grade/convai2-grade_processed",
        "annot_path": "data/DSTC_11_Track_4/metadata/dev/convai2-grade/convai2-grade_eval_zh_es_pa.json",
        "annotations": ["relevance"],
        "aggregation": np.mean,
    },
    "dailydialog-grade": {
        "path": "data/DSTC_11_Track_4/metadata/dev/dailydialog-grade/dailydialog-grade_processed",
        "annot_path": "data/DSTC_11_Track_4/metadata/dev/dailydialog-grade/dailydialog-grade_eval_zh_es_pa.json",
        "annotations": ["relevance"],
        "aggregation": np.mean,
    },
    "dailydialog-gupta": {
        "path": "data/DSTC_11_Track_4/metadata/dev/dailydialog-gupta/dailydialog-gupta_processed",
        "annot_path": "data/DSTC_11_Track_4/metadata/dev/dailydialog-gupta/dailydialog-gupta_eval_zh_es_pa.json",
        "annotations": ["overall"],
        "aggregation": lambda x: x[0],
    },
    "dailydialog-zhao": {
        "path": "data/DSTC_11_Track_4/metadata/dev/dailydialog-zhao/dailydialog-zhao_processed",
        "annot_path": "data/DSTC_11_Track_4/metadata/dev/dailydialog-zhao/dailydialog-zhao_eval_zh_es_pa.json",
        "num_references": 1,
        "annotations": ["content", "grammar", "appropriateness", "relevance"],
        "aggregation": np.mean,
    },
    "dstc7": {
        "path": "data/DSTC_11_Track_4/metadata/dev/dstc7/dstc7_processed",
        "annot_path": "data/DSTC_11_Track_4/metadata/dev/dstc7/dstc7_eval_zh_es_pa.json",
        "num_references": 1,
        "annotations": ["informativeness", "overall", "relevance"],
        "aggregation": np.mean,
    },
    "empathetic-grade": {
        "path": "data/DSTC_11_Track_4/metadata/dev/empathetic-grade/empathetic-grade_processed",
        "annot_path": "data/DSTC_11_Track_4/metadata/dev/empathetic-grade/empathetic-grade_eval_zh_es_pa.json",
        "annotations": ["relevance"],
        "aggregation": np.mean,
    },
    "fed-dial": {
        "path": "data/DSTC_11_Track_4/metadata/dev/fed-dial/fed-dial_processed",
        "annot_path": "data/DSTC_11_Track_4/metadata/dev/fed-dial/fed-dial_eval_zh_es_pa.json",
        "annotations": [
            "Coherent",
            "Error recovery",
            "Consistent",
            "Diverse",
            "Depth",
            "Likeable",
            "Understanding",
            "Flexible",
            "Informative",
            "Inquisitive",
            "Overall",
        ],
        "aggregation": np.mean,
    },
    "fed-turn": {
        "path": "data/DSTC_11_Track_4/metadata/dev/fed-turn/fed-turn_processed",
        "annot_path": "data/DSTC_11_Track_4/metadata/dev/fed-turn/fed-turn_eval_zh_es_pa.json",
        "annotations": [
            "Correct",
            "Engaging",
            "Fluent",
            "Interesting",
            "Overall",
            "Relevant",
            "Semantically appropriate",
            "Specific",
            "Understandable",
        ],
        "aggregation": np.mean,
    },
    "humod": {
        "path": "data/DSTC_11_Track_4/metadata/dev/humod/humod_processed",
        "annot_path": "data/DSTC_11_Track_4/metadata/dev/humod/humod_eval_zh_es_pa.json",
        "num_references": 3,
        "annotations": ["relevance", "language_usage"],
        "aggregation": np.mean,
    },
    "persona-see": {
        "path": "data/DSTC_11_Track_4/metadata/dev/persona-see/persona-see_processed",
        "annot_path": "data/DSTC_11_Track_4/metadata/dev/persona-see/persona-see_eval_zh_es_pa.json",
        "annotations": [
            "avoid_rep",
            "enjoy",
            "fluency",
            "inquisitive",
            "interest",
            "listen",
            "make_sense",
            "persona_guess",
            "turing",
        ],
        "aggregation": lambda x: x[0],
    },
    "persona-usr": {
        "path": "data/DSTC_11_Track_4/metadata/dev/persona-usr/persona-usr_processed",
        "annot_path": "data/DSTC_11_Track_4/metadata/dev/persona-usr/persona-usr_eval_zh_es_pa.json",
        "num_references": 1,
        "annotations": [
            "Understandable",
            "Natural",
            "Maintains Context",
            "Engaging",
            "Uses Knowledge",
            "Overall",
        ],
        "aggregation": np.mean,
    },
    "persona-zhao": {
        "path": "data/DSTC_11_Track_4/metadata/dev/persona-zhao/persona-zhao_processed",
        "annot_path": "data/DSTC_11_Track_4/metadata/dev/persona-zhao/persona-zhao_eval_zh_es_pa.json",
        "num_references": 1,
        "annotations": ["appropriateness"],
        "aggregation": np.mean,
    },
    "topical-usr": {
        "path": "data/DSTC_11_Track_4/metadata/dev/topical-usr/topical-usr_processed",
        "annot_path": "data/DSTC_11_Track_4/metadata/dev/topical-usr/topical-usr_eval_zh_es_pa.json",
        "annotations": [
            "Understandable",
            "Natural",
            "Maintains Context",
            "Engaging",
            "Uses Knowledge",
            "Overall",
        ],
        "aggregation": np.mean,
    },
}

gpt_metrics = {
    "gpt-overall": (
        "Given the Context, evaluate from 1-5 the Response in terms of Appropriateness. Provide the score and nothing else.",
        "Evaluate the following dialogue from 1-5 in terms of Overall Quality. Provide the score and nothing else.",
    ),
    "gpt-relevant": (
        "Given the Context, evaluate from 1-5 the Response in terms of Relevance. Provide the score and nothing else.",
        "Evaluate the following dialogue from 1-5 in terms of Coherence. Provide the score and nothing else.",
    ),
    "gpt-engaging": (
        "Given the Context, evaluate from 1-5 the Response in terms of Content Richness. Provide the score and nothing else.",
        "Evaluate the following dialogue from 1-5 in terms of Likeability. Provide the score and nothing else.",
    ),
    "gpt-content": (
        "Evaluate from 1-5 the Response in terms of Grammatical Correctness. Provide the score and nothing else.",
        "Evaluate the following dialogue from 1-5 in terms of Informativeness. Provide the score and nothing else.",
    ),
}

logger = logging.getLogger()


def preprocess_ctxres(examples):
    args = (examples["ctx"], examples["res"])
    result = tokenizer(
        *args, padding="max_length", max_length=MAX_LENGTH, truncation=True
    )
    return result


def preprocess_res(examples):
    args = (examples["res"],)
    result = tokenizer(
        *args, padding="max_length", max_length=MAX_LENGTH, truncation=True
    )
    return result


preprocess_funcs = {
    "vsp": preprocess_res,
    "nsp": preprocess_ctxres,
    "engagement": preprocess_ctxres,
}


if __name__ == "__main__":
    dt = datetime.now()
    set_verbosity_info()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = ArgumentParser(
        prog="DSTC11 predictor",
        description="Outputs predictions for DSTC11 shared task",
    )

    parser.add_argument(
        "--options",
        nargs="*",
        choices=[
            "all",
            "vsp",
            "nsp",
            "mlm",
            "engagement",
            "gpt-overall",
            "gpt-relevant",
            "gpt-engaging",
            "gpt-content",
        ],
        default="all",
    )
    parser.add_argument(
        "--langs",
        nargs="*",
        choices=["all", "en", "es", "zh", "pa"],
        default="all",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=dev.keys(),
        default="all",
    )
    parser.add_argument("--predict", action=BooleanOptionalAction, default=False)
    parser.add_argument("--eval", action=BooleanOptionalAction, default=False)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--pretrained_model", type=str, default="xlm-roberta-large")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=f"logs/{dt.strftime('%d-%m-%Y_%H.%M.%S')}",
    )

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=f"{args.output_dir}/{dt.strftime('%d-%m-%Y_%H.%M.%S')}.log",
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if "all" in args.options:
        submetrics = [
            "vsp",
            "nsp",
            "mlm",
            "engagement",
            "gpt-overall",
            "gpt-relevant",
            "gpt-engaging",
            "gpt-content",
        ]
    else:
        submetrics = args.options

    if "all" in args.datasets:
        datasets = dev.keys()
    else:
        datasets = args.datasets

    if "all" in args.langs:
        langs = [
            "en",
            "es",
            "zh",
            "pa",
        ]
    else:
        langs = args.langs

    if args.predict:
        logger.info("Predicting...")
        training_args = TrainingArguments(
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            do_predict=True,
            output_dir="exp/out",
            label_names=["labels"],
        )
        config = AutoConfig.from_pretrained(
            args.pretrained_model,
            num_labels=1,
        )
        special_token_dict = {
            "speaker1_token": "<speaker1>",
            "speaker2_token": "<speaker2>",
        }

        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model,
            use_fast=True,
        )
        tokenizer.add_tokens(list(special_token_dict.values()))
        predictions = dict()
        for submetric in submetrics:
            predictions[submetric] = dict()
            if submetric in ["nsp", "vsp", "engagement"]:
                for model_name, model_path in models[submetric].items():
                    predictions[submetric][model_name] = dict()
                    model = AutoModelWithHeads.from_pretrained(
                        models[submetric][model_name],
                        config=models[submetric][model_name],
                    )
                    model.resize_token_embeddings(len(tokenizer))
                    model.to(device)

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=None,
                        eval_dataset=None,
                        tokenizer=tokenizer,
                        data_collator=default_data_collator,
                    )

                    for dataset_name in datasets:
                        dataset_data = dev[dataset_name]
                        predictions[submetric][model_name][dataset_name] = dict()
                        for lang in langs:
                            data_path = dataset_data["path"] + f"_{lang}.csv"
                            ds = load_dataset(
                                "csv",
                                data_files={"test": data_path},
                                download_mode="force_redownload",
                            )
                            ds = ds.map(preprocess_funcs[submetric], batched=True)
                            preds, _, metadata = trainer.predict(
                                test_dataset=ds["test"]
                            )
                            predictions[submetric][model_name][dataset_name][
                                lang
                            ] = preds.tolist()
                            path = args.output_dir + f"/{submetric}/{model_name}/"
                            Path(path).mkdir(parents=True, exist_ok=True)
                            with open(
                                path + f"{dataset_name}_{lang}.json",
                                "w",
                            ) as outfile:
                                json.dump(preds.squeeze().tolist(), outfile)

            elif submetric == "mlm":
                model = XLMRobertaForMaskedLM.from_pretrained(args.pretrained_model)
                model.to(device)
                mlm_predictor = MLM_predictor(model, tokenizer, device)

                for dataset_name in datasets:
                    dataset_data = dev[dataset_name]
                    predictions[submetric][dataset_name] = dict()
                    for lang in langs:
                        data_path = dataset_data["path"] + f"_{lang}.csv"
                        data_test = pd.read_csv(data_path)

                        preds = [
                            mlm_predictor.predict(x[10:]) for x in tqdm(data_test.res)
                        ]
                        predictions[submetric][dataset_name][lang] = preds
                        path = args.output_dir + f"/{submetric}/"
                        Path(path).mkdir(parents=True, exist_ok=True)
                        with open(
                            path + f"/{dataset_name}_{lang}.json", "w"
                        ) as outfile:
                            json.dump(preds, outfile)

            elif submetric in [
                "gpt-overall",
                "gpt-relevant",
                "gpt-engaging",
                "gpt-content",
            ]:
                for dataset_name in datasets:
                    if dataset_name in DIAL_LVL_DATASETS:
                        prompt = gpt_metrics[submetric][1]
                    else:
                        prompt = gpt_metrics[submetric][0]

                    dataset_data = dev[dataset_name]
                    predictions[submetric][dataset_name] = dict()
                    for lang in langs:
                        data_path = dataset_data["annot_path"]
                        data_test = json.load(open(data_path))
                        preds = []
                        for i in tqdm(range(0, len(data_test))):
                            lang_key = "" if lang == "en" else f"_{lang}"
                            ctx = data_test[i]["context" + lang_key]
                            res = data_test[i]["response" + lang_key]
                            # fallback if paraphrase returns None
                            if (ctx == "None" or ctx == None) and lang == "pa":
                                ctx = data_test[i]["context"]
                            if (res == "None" or ctx == None) and lang == "pa":
                                res = data_test[i]["response"]

                            if dataset_name in DIAL_LVL_DATASETS:
                                text_prompt = ctx + "\n" + res
                            else:
                                text_prompt = f"Context:{ctx}\nResponse:{res}"

                            gpt_score = -1
                            score_backoff = 0
                            while gpt_score == -1 and score_backoff < 2:
                                completion = completions_with_backoff(
                                    model="gpt-3.5-turbo",
                                    temperature=0.0,
                                    messages=[
                                        {"role": "system", "content": prompt},
                                        {"role": "user", "content": text_prompt},
                                    ],
                                )
                                gpt_score = completion_to_score(
                                    completion.choices[0].message.content
                                )
                                score_backoff += 1

                            if gpt_score != -1:
                                preds.append(gpt_score)
                            else:
                                logger.info(
                                    f"\nGPT Error! - Prompt: {prompt} | {text_prompt} - Output: {completion.choices[0].message.content} - Datapoint {i} | {lang} | {dataset_name} - defaulting to 3.\n"
                                )
                                preds.append(3)
                        predictions[submetric][dataset_name][lang] = preds
                        path = args.output_dir + f"/{submetric}/"
                        Path(path).mkdir(parents=True, exist_ok=True)
                        with open(
                            path + f"{dataset_name}_{lang}.json",
                            "w",
                        ) as outfile:
                            json.dump(preds, outfile)

        dev_normalize(predictions, args.output_dir, DIAL_LVL_DATASETS, dev)

        with open(args.output_dir + "/all.json", "w") as outfile:
            json.dump(predictions, outfile)

    if args.eval:
        logger.info("Evaluating...")
        if not args.predict:
            if args.eval_file:
                with open(args.eval_file, "r") as outfile:
                    predictions = json.load(outfile)
            else:
                raise FileNotFoundError(
                    "No prediction file provided. Either provide file using --eval_file or run --predict."
                )

        results = dict()
        weights = dict()
        human_scores = dict()
        for dataset_name in datasets:
            dataset_data = dev[dataset_name]
            results[dataset_name] = dict()
            weights[dataset_name] = dict()
            with open(dataset_data["annot_path"], "r") as f:
                dataset_df = pd.json_normalize(json.load(f))
            dataset_df = normalize_df(dataset_name, dataset_df, dev)
            annotations = ["annotations." + _ for _ in dev[dataset_name]["annotations"]]
            human_scores[dataset_name] = dict()
            for k in annotations:
                human_scores[dataset_name][k] = list(dataset_df[k])
            for lang in langs:
                results[dataset_name][lang] = dict()
                weights[dataset_name][lang] = dict()
                for quality in human_scores[dataset_name]:
                    labels = human_scores[dataset_name][quality]
                    results[dataset_name][lang][quality] = dict()
                    pearsons = []
                    p_values = []
                    for submetric in submetrics:
                        results[dataset_name][lang][quality][submetric] = dict()
                        if submetric in ["nsp", "vsp", "engagement"]:
                            avg = []
                            for model in predictions[submetric].keys():
                                preds = predictions[submetric][model][dataset_name][
                                    lang
                                ]
                                avg.append(preds)
                                rho, p_value = stats.spearmanr(
                                    preds, labels, nan_policy="omit"
                                )
                                results[dataset_name][lang][quality][submetric][
                                    model
                                ] = rho

                            rho, p_value = stats.spearmanr(
                                labels, np.nanmean((avg), axis=0), nan_policy="omit"
                            )
                            results[dataset_name][lang][quality][submetric][
                                "mean"
                            ] = rho

                            x_train, _, y_train, _ = train_test_split(
                                preds, labels, test_size=0.7
                            )
                            rho, p_value = stats.spearmanr(
                                y_train, x_train, nan_policy="omit"
                            )
                            pearsons.append(rho)
                            p_values.append(p_value)
                        else:
                            preds = predictions[submetric][dataset_name][lang]
                            rho, p_value = stats.spearmanr(
                                labels, preds, nan_policy="omit"
                            )
                            results[dataset_name][lang][quality][submetric] = rho

                            x_train, _, y_train, _ = train_test_split(
                                preds, labels, test_size=0.6
                            )
                            rho, p_value = stats.spearmanr(
                                y_train, x_train, nan_policy="omit"
                            )
                            pearsons.append(rho)
                            p_values.append(p_value)

                    # p_values > 0.05 pearsons <0 unless mlm
                    negatives = np.array(pearsons) < 0
                    negatives[2] = False  # N-MLM
                    pearsons_masked = np.ma.masked_where(
                        np.logical_or(np.array(p_values) >= 0.05, negatives),
                        pearsons,
                    )
                    if np.ma.count_masked(pearsons_masked) == len(pearsons_masked):
                        pearsons_masked = np.ma.masked_where(negatives, pearsons)
                    pearsons_masked = pearsons_masked.filled(0.0)
                    weights[dataset_name][lang][quality] = list(
                        pearsons_masked
                        * np.abs(pearsons_masked)
                        / sum(pearsons_masked**2)
                    )

        with open(args.output_dir + "/results.json", "w") as outfile:
            json.dump(results, outfile)

        with open(args.output_dir + "/weights.json", "w") as outfile:
            json.dump(weights, outfile)

        logger.info("Done calculating weights.")

        final_preds = dict()
        final_results = dict()
        for dataset_name in datasets:
            final_preds[dataset_name] = dict()
            final_results[dataset_name] = dict()
            # loop to get all submetrics for given dataset_lang
            for lang in langs:
                final_preds[dataset_name][lang] = list()
                for submetric in submetrics:
                    if submetric in ["nsp", "vsp", "engagement"]:
                        avg = []
                        for model in predictions[submetric].keys():
                            avg.append(
                                predictions[submetric][model][dataset_name][lang]
                            )
                        final_preds[dataset_name][lang].append(np.mean((avg), axis=0))
                    else:
                        final_preds[dataset_name][lang].append(
                            predictions[submetric][dataset_name][lang]
                        )

            for lang in langs:
                final_results[dataset_name][lang] = dict()
                for quality in human_scores[dataset_name]:
                    fused_prediction = np.dot(
                        weights[dataset_name][lang][quality],
                        (final_preds[dataset_name][lang]),
                    )
                    rho, p_value = stats.spearmanr(
                        human_scores[dataset_name][quality],
                        fused_prediction,
                        nan_policy="omit",
                    )
                    final_results[dataset_name][lang][quality] = rho

        with open(args.output_dir + "/results_final.json", "w") as outfile:
            json.dump(final_results, outfile)
