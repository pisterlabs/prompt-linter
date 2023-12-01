import hashlib
import json
import os
from typing import Any

import evaluate
import numpy as np
import wandb
from datasets import load_dataset
from langchain.text_splitter import TokenTextSplitter
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer

WANDB_LOG_MODEL = "end"


def get_id_for_dict(config_dict: dict[str, Any]) -> str:
    """This function creates a unique hash
    based on the initial config dictionary
    that is used to label the model."""

    unique_str = "".join(
        f"'{key}':'{value}';" for key, value in sorted(config_dict.items())
    )
    return hashlib.sha1(unique_str.encode("utf-8")).hexdigest()[:5]


def compute_metrics(eval_pred):
    metric1 = evaluate.load("precision")
    metric2 = evaluate.load("recall")
    metric3 = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision = metric1.compute(
        predictions=predictions, references=labels, average="macro"
    )["precision"]
    recall = metric2.compute(
        predictions=predictions, references=labels, average="macro"
    )["recall"]
    f1 = metric3.compute(predictions=predictions, references=labels)["f1"]
    f1_macro = metric3.compute(
        predictions=predictions, references=labels, average="macro"
    )["f1"]
    return {"precision": precision, "recall": recall, "f1": f1, "f1_macro": f1_macro}


def tokenize_function(examples):
    """Tokenize the text. We want the review to be the first sentence, then SEP token and then the paper."""
    reviews = [
        f"{review_title} {review_criteria}"
        for review_title, review_criteria in zip(
            examples["review_title"], examples["review_criteria"]
        )
    ]
    last_n = 3000
    reviews = [review_splitter.split_text(review)[0] for review in reviews]
    papers = [
        f"{title} {abstract} {main_text[-last_n:]}"
        for title, abstract, main_text in zip(
            examples["title"], examples["abstract"], examples["main_text"]
        )
    ]

    input_text = [
        review + tokenizer.sep_token + paper for review, paper in zip(reviews, papers)
    ]
    return tokenizer(input_text, padding="max_length", truncation=True)


if __name__ == "__main__":
    outpath = "../../data/processed/classifiers/"

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    models = {
        "allenai/longformer-base-4096": 4096,
        "yikuan8/Clinical-BigBird": 4096,
        "yikuan8/Clinical-Longformer": 4096,
        # "google/bigbird-pegasus-large-pubmed": 4096,
        "google/bigbird-roberta-base": 4096,
    }

    per_device_train_batch_size = 1
    per_device_eval_batch_size = 1
    gradient_accumulation_steps = 8
    learning_rate = 1e-5
    weight_decay = 0.01
    adam_beta1 = 0.9
    num_train_epochs = 10

    dataset = load_dataset(
        "../../csmed/datasets/datasets/csmed_ft", name="csmed_ft_all_source"
    )
    for learning_rate in [5e-5, 2e-5, 1e-5, 1e-6, 1e-7]:
        for model_name, context_size in models.items():
            wandb.init(
                project="full_text_classification",
                name=f"{model_name}_{learning_rate}_{weight_decay}_last3000",
                config={
                    "model_name": model_name,
                    "context_size": context_size,
                    "per_device_train_batch_size": per_device_train_batch_size,
                    "per_device_eval_batch_size": per_device_eval_batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "adam_beta1": adam_beta1,
                    "num_train_epochs": num_train_epochs,
                },
            )

            print(f"Using model {model_name}, context size {context_size}")

            review_splitter = TokenTextSplitter(
                chunk_size=context_size // 2,
                chunk_overlap=0,
                model_name="gpt-3.5-turbo-0301",
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            wandb.watch(model)

            tokenized_datasets = dataset.map(tokenize_function, batched=True)

            training_args = TrainingArguments(
                output_dir="test_trainer",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                per_device_train_batch_size=wandb.config.per_device_train_batch_size,
                per_device_eval_batch_size=wandb.config.per_device_eval_batch_size,
                gradient_accumulation_steps=wandb.config.gradient_accumulation_steps,
                learning_rate=wandb.config.learning_rate,
                weight_decay=wandb.config.weight_decay,
                adam_beta1=wandb.config.adam_beta1,
                num_train_epochs=wandb.config.num_train_epochs,
                report_to=["wandb"],
                logging_steps=1,
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1",
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                compute_metrics=compute_metrics,
            )

            trainer.train()

            trainer.evaluate()

            for test_data in ["validation", "sample", "test"]:
                predictions = trainer.predict(
                    tokenized_datasets[test_data],
                )
                wandb.log(
                    {
                        f"{test_data}_precision": predictions.metrics["test_precision"],
                        f"{test_data}_recall": predictions.metrics["test_recall"],
                        f"{test_data}_f1": predictions.metrics["test_f1"],
                        f"{test_data}_f1_macro": predictions.metrics["test_f1_macro"],
                        f"{test_data}_loss": predictions.metrics["test_loss"],
                    }
                )

                with open(
                    f"{outpath}/{'_'.join(model_name.split('/'))}_{test_data}.json", "w"
                ) as fp:
                    json.dump(predictions.predictions.tolist(), fp)
            wandb.finish()
