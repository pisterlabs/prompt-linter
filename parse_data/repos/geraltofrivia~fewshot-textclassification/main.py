"""
    First we're going to just follow the thing and play with the metrics once done

"""
import json
import os
import random
import warnings
from pathlib import Path

import click
import langchain
import numpy as np
import numpy.random
import torch
from datasets import load_dataset, DatasetDict
from datasets.features.features import ClassLabel
from langchain import HuggingFaceHub, PromptTemplate, FewShotPromptTemplate, LLMChain
from langchain.cache import InMemoryCache
from langchain.prompts.example_selector import LengthBasedExampleSelector
from mytorch.utils.goodies import FancyDict
from sentence_transformers.losses import CosineSimilarityLoss
from setfit.trainer import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from transformers import AutoConfig

from overrides import CustomTrainer, CustomModel

langchain.llm_cache = InMemoryCache()


random.seed(42)
torch.manual_seed(42)
numpy.random.seed(42)


# @contextmanager
# def suppress_stdout_stderr():
#     """A context manager that redirects stdout and stderr to devnull"""
#     with open(devnull, "w") as fnull:
#         with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
#             yield (err, out)


def sanitize_dataset(dataset: DatasetDict):
    """make sure the has everything we need: text label, label_dict"""
    if "label_text" in dataset["train"].column_names:
        return dataset

    # check if label field has this information and if so, make it into a column
    elif isinstance(dataset["train"].features["label"], ClassLabel) and hasattr(
        dataset["train"].features["label"], "names"
    ):
        id_to_label = {_id: _label for _id, _label in enumerate(dataset["train"].features["label"].names)}
        for k, v in dataset.items():
            if k == "unsupervised":
                continue
            dataset[k] = dataset[k].add_column("label_text", [id_to_label[label] for label in dataset[k]["label"]])
        return dataset

    elif len(set(dataset["train"]["label"])) == 2:
        # its a binary task; assume 0 as neg; 1 as pos
        id_to_label = {0: "negative", 1: "positive"}
        for k, v in dataset.items():
            dataset[k] = dataset[k].add_column("label_text", [id_to_label[label] for label in dataset[k]["label"]])
        return dataset

    else:
        raise ValueError("The given dataset seems incompatible for the task.")


def case0(
    dataset: DatasetDict,
    seed: int,
    num_sents: int,
    num_epochs: int,
    num_epochs_finetune: int,
    num_iters: int,
    batch_size: int,
    test_on_test: bool = False,
    *args,
    **kwargs,
) -> dict:
    """
    Do exactly what the blogpost does

    Get SetFit model (with ST and LogClf)
    # Step 1:
    Create pairs
    Fine tune ST on faux task (cosine thing)
    Fit Log reg on main task

    # Step 2:
    Run model to classify on main task
    Report Accuracy
    """
    model = CustomModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

    # Sample num_sents from the dataset. Divide them in 80/20
    train_ds = dataset["train"].shuffle(seed=seed).select(range(num_sents))
    if test_on_test:
        test_ds = dataset["test"]
    else:
        train_ds, test_ds = train_ds.select(range(int(len(train_ds) * 0.8))), train_ds.select(
            range(int(len(train_ds) * 0.8), len(train_ds))
        )

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        loss_class=CosineSimilarityLoss,
        batch_size=batch_size,
        num_iterations=num_iters,  # Number of text S to generate for contrastive learning
        num_epochs=num_epochs,  # Number of epochs to use for contrastive learning
    )

    # Fit the ST on the cosinesim task; Fit the entire thing on the main task
    trainer.train(num_epochs=num_epochs, num_epochs_finetune=num_epochs_finetune)

    metrics = trainer.evaluate()
    # print(metrics)
    return metrics


def case1(
    dataset: DatasetDict,
    seed: int,
    num_sents: int,
    num_epochs: int,
    num_epochs_finetune: int,
    num_iters: int,
    batch_size: int,
    test_on_test: bool = False,
    *args,
    **kwargs,
) -> dict:
    """
    This is regular fine-tuning. Noisy.
    Skip ST Finetuning; Slap a classifier and train the thing together.

    Get SetFit model (with ST and DenseHead).
    # Step 1
    Create pairs
    DONT Fine tune ST on faux task (Cosine)
    Fit DenseHead + ST on main task

    # Step 2
    Run model to classify on main task
    Report Accuracy
    """
    num_classes = max(dataset["train"]["label"]) + 1
    num_classes = max(dataset["train"]["label"]) + 1
    model = CustomModel.from_pretrained(
        "sentence-transformers/paraphrase-mpnet-base-v2",
        use_differentiable_head=True,
        head_params={"out_features": num_classes},
    )
    # Sample num_sents from the dataset. Divide them in 80/20
    train_ds = dataset["train"].shuffle(seed=seed).select(range(num_sents))
    if test_on_test:
        test_ds = dataset["test"]
    else:
        train_ds, test_ds = train_ds.select(range(int(len(train_ds) * 0.8))), train_ds.select(
            range(int(len(train_ds) * 0.8), len(train_ds))
        )

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        loss_class=CosineSimilarityLoss,
        batch_size=batch_size,
        num_iterations=num_iters,  # Number of text pairs to generate for contrastive learning
        num_epochs=num_epochs,  # Number of epochs to use for contrastive learning
    )
    # trainer.se

    # Fit the ST on the cosinesim task; Fit the entire thing on the main task
    # trainer.train(num_epochs=num_epochs, num_epochs_finetune=num_epochs_finetune) #, do_fitclf_trainencoder=False)
    trainer.train(
        num_epochs=num_epochs,
        num_epochs_finetune=num_epochs_finetune,
        do_fitclf_trainencoder=True,
        do_finetune=False,
    )

    metrics = trainer.evaluate()
    # print(metrics)
    return metrics


def case2(
    dataset: DatasetDict,
    seed: int,
    num_sents: int,
    num_epochs: int,
    num_epochs_finetune: int,
    num_iters: int,
    batch_size: int,
    test_on_test: bool = False,
    *args,
    **kwargs,
):
    """
    Get SetFit model (ST + LogClf Head)

    # Step 1
    Do not fine-tune ST on faux task (Cosine)
    Just fit LogClf on the main task (freeze body)

    # Step 2
    Run model to classify on main task
    Report Accuracy
    """
    model = CustomModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2", use_differentiable_head=False)

    # Sample num_sents from the dataset. Divide them in 80/20
    train_ds = dataset["train"].shuffle(seed=seed).select(range(num_sents))
    if test_on_test:
        test_ds = dataset["test"]
    else:
        train_ds, test_ds = train_ds.select(range(int(len(train_ds) * 0.8))), train_ds.select(
            range(int(len(train_ds) * 0.8), len(train_ds))
        )

    # Freeze the head (so we never train/finetune ST)
    trainer = CustomTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        loss_class=CosineSimilarityLoss,
        batch_size=batch_size,
        num_iterations=num_iters,  # Number of text pairs to generate for contrastive learning
        num_epochs=num_epochs,  # Number of epochs to use for contrastive learning
    )

    trainer.train(
        num_epochs=num_epochs,
        num_epochs_finetune=num_epochs_finetune,
        do_finetune=False,
    )

    metrics = trainer.evaluate()
    # print(metrics)
    return metrics


def case3(
    dataset: DatasetDict,
    seed: int,
    num_sents: int,
    batch_size: int,
    test_on_test: bool = False,
    *args,
    **kwargs,
):
    """
    Uses langchain to throw questions to HF model Flan t5 xl.
    """
    # First figure out the length of the model
    config = AutoConfig.from_pretrained("google/flan-t5-xl")
    max_len = config.n_positions

    # Read HuggingFace API key
    try:
        with (Path(".") / "hf_token.key").open("r") as f:
            hf_token_key = f.read().strip()
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token_key
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No HuggingFace API key found at {(Path('.') / 'hf_token.key').absolute()}"
            "You need to generate yours at https://huggingface.co/settings/tokens"
            "and paste it in this file."
        )

    # # initialize Hub LLM
    hub_llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature": 1e-10})

    # label_to_id = {"negative": 0, "positive": 1}
    # id_to_label = {v: k for k, v in label_to_id.items()}

    # Go through the dataset, generate train and testset
    # Sample num_sents from the dataset. Divide them in 80/20
    train_ds = dataset["train"].shuffle(seed=seed).select(range(num_sents))
    if test_on_test:
        test_ds = dataset["test"]
    else:
        train_ds, test_ds = train_ds.select(range(int(len(train_ds) * 0.8))), train_ds.select(
            range(int(len(train_ds) * 0.8), len(train_ds))
        )

    """ Prompt stuff """
    # create a example template
    example_template = """
    Review: {query}
    Sentiment: {answer}
    """
    # create a prompt example from above template
    example_prompt = PromptTemplate(input_variables=["query", "answer"], template=example_template)
    examples = [{"query": x["text"], "answer": x["label_text"]} for x in train_ds]
    prefix = f"""Classify into {' or '.join(set(dataset['train']['label_text']))}. Here are some examples: """
    suffix = """
    Review: {query}
    Sentiment: 
    """
    # We'll use the `LengthBasedExampleSelector` to select the examples.
    example_selector = LengthBasedExampleSelector(
        # These are the examples is has available to choose from.
        examples=examples,
        # This is the PromptTemplate being used to format the examples.
        example_prompt=example_prompt,
        # This is the maximum length that the formatted examples should be.
        # Length is measured by the get_text_length function below.
        max_length=max_len,
    )

    # now create the few shot prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n\n",
    )
    llm_chain = LLMChain(prompt=few_shot_prompt_template, llm=hub_llm)

    # Bug found: can't use dataloader if dataset has datetime
    if "date" in test_ds.column_names:
        test_ds = test_ds.remove_columns("date")

    score = []
    for batch in tqdm(DataLoader(test_ds, batch_size=batch_size)):
        texts = [{"query": instance} for instance in batch["text"]]

        answers = None
        answers = llm_chain.generate(texts)
        # while True:
        #     try:
        #         answers = llm_chain.generate(texts)
        #         break
        #     except ValueError as e:
        #         print(e) # so you can see the error.
        #         if input("Enter `q` to stop, else we keep trying to send this request again.") == 'q' or no_retry:
        #             break
        #         else:
        #             continue

        # Iterate through answers and labels
        for i, generation in enumerate(answers.generations):
            answer = generation[0].text.strip().lower()
            try:
                if batch["label_text"][i] == answer:
                    score.append(1)
                else:
                    score.append(0)
            except KeyError:
                warnings.warn(f"The answer to {i}th element is `{answer}`.")
                score.append(0)

    return {"accuracy": np.mean(score)}


def merge_metrics(list_of_metrics):
    pooled = {}
    for metrics in list_of_metrics:
        for k, v in metrics.items():
            pooled.setdefault(k, []).append(v)

    return pooled


def normalize_dataset(dataset: DatasetDict):
    """Check if text and label exist or not. Further if label_text doesn't exist makes 0 as neg 1 as pos"""


@click.command()
@click.option(
    "--dataset-name",
    "-d",
    type=str,
    default="SetFit/SentEval-CR",
    help=(
        "The name of the dataset as it appears on the HuggingFace hub "
        "e.g. SetFit/SentEval-CR | SetFit/bbc-news | SetFit/enron_spam ... "
    ),
)
@click.option(
    "--case",
    "-c",
    type=int,
    required=True,
    help=(
        "0, 1, 2, or 3: which experiment are we running. See readme or docstrings to know more but briefly: "
        "**0**: SentTF -> Constrastive Pretrain -> +LogReg on task. "
        "**1**: SentTF -> +Dense on task. "
        "**2**: SentTF -> +LogReg on task. "
        "**3**: FewShotPrompting based Clf over Flan-t5-xl"
    ),
)
@click.option(
    "--repeat",
    "-r",
    type=int,
    default=1,
    help="The number of times we should run the entire experiment (changing the seed).",
)
@click.option("--batch-size", "-bs", type=int, default=16, help="... you know what it is.")
@click.option("--num-sents", "-ns", type=int, default=64, help="Size of our train set. Set short values (under 100)")
@click.option(
    "--num-epochs",
    "-e",
    type=int,
    default=1,
    help="Epochs for fitting Clf+SentTF on the main (classification) task.",
)
@click.option(
    "--num-epochs-finetune",
    "-eft",
    type=int,
    default=1,
    help="Epochs for both contrastive pretraining of SentTF.",
)
@click.option(
    "--num-iters",
    "-ni",
    type=int,
    default=20,
    help="Number of text pairs to generate for contrastive learning. Values above 20 can get expensive to train.",
)
@click.option(
    "--test-on-test",
    "-tot",
    is_flag=True,
    default=False,
    help="If true, we report metrics on testset. If not, on a 20% split of train set. Off by default.",
)
@click.option(
    "--full-test",
    "-ft",
    is_flag=True,
    default=False,
    help=(
        "We truncate the testset of every dataset to have 100 instances. "
        "If you know what you're doing, you can test on the full dataset."
        "NOTE that if you're running this in case 3 you should probably be a premium member and not be paying per use."
    ),
)
def run(
    dataset_name: str,
    repeat: int,
    batch_size: int,
    num_epochs: int,
    num_epochs_finetune: int,
    num_sents: int,
    num_iters: int,
    case: int,
    full_test: int,
    test_on_test: bool,
):
    try:
        fname = globals()[f"case{case}"]
    except KeyError:
        raise ValueError(f"No function called case{case}")

    if repeat < 1:
        raise ValueError("Repeats must be greater than 0")

    config = FancyDict(
        **{
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "num_epochs_finetune": num_epochs_finetune,
            "num_sents": num_sents,
            "num_iters": num_iters,
            "test_on_test": test_on_test,
        }
    )

    if case == 3:
        repeat = 1
        warnings.warn(f"On case 3 i.e. prompting LLMs, we do not repeat to respect the rate limits.")

    if not dataset_name.startswith("SetFit/"):
        warnings.warn(f"We expect the dataset to have these fields `text`, `label` and `label_text`.")

    metrics = []
    for _ in trange(repeat, desc=f"Case: {case} | DS: {dataset_name}"):
        seed = random.randint(0, 200)
        set_seed(seed)

        # Pull the dataset
        dataset = load_dataset(dataset_name)

        # Make sure the dataset is consistent (has label, text, and label_text fields)
        dataset = sanitize_dataset(dataset)

        # Going to truncate the testsets to be 100 (unless flagged otherwise)
        if (len(dataset["test"]) > 100) and not full_test:
            dataset["test"] = dataset["test"].shuffle(seed).select(range(100))

        # Run the case (based on the case specified in args)
        metric = fname(dataset, seed=seed, **config)
        metrics.append(metric)

    print(f"---------- FINALLY, over {repeat} runs, with case {case} and dataset {dataset_name} -----------")
    metrics = merge_metrics(metrics)
    print({k: f"{np.mean(v):.3f} +- {np.std(v):.3f}" for k, v in metrics.items()})
    metrics["config"] = config

    # Dump the summaries to disk
    dumpdir = Path(f"summaries") / f"{dataset_name.split('/')[-1]}_{num_sents}"
    dumpdir.mkdir(parents=True, exist_ok=True)
    with (dumpdir / f"case_{case}.json").open("w+") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    run()
    # config = FancyDict(**{"test_on_test": True, "num_epochs": 1, "batch_size": 16})
    # test_on_test = True
    #
    # print("------------")
    # print("---Case 0---")
    # # dataset = load_dataset("SetFit/SentEval-CR")
    # # case0(dataset, 42, 50, **config)
    # print("------------")
    # print("---Case 1---")
    # dataset = load_dataset()
    # case1(dataset, 42, 50, **config)
    # print("------------")
    # print("---Case 1---")
    # # dataset = load_dataset("SetFit/SentEval-CR")
    # # case2(dataset, 42, 50, **config)
