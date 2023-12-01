import os
import time
import json
import torch
import random
import openai
import argparse
from tqdm import tqdm
from typing import List
from dotenv import load_dotenv

from sectors.utils.trie import Trie
from sectors.utils.models import create_model
from sectors.utils.evaluation import sector_report
from sectors.utils.dataset import MultiLabelT2TDataset
from sectors.config import DATA_DIR, BASE_DIR


load_dotenv()
openai.organization = os.getenv("OPENAI_ORGANIZATION_ID")
openai.api_key = os.getenv("OPENAI_SECRET_KEY")
random.seed(42)


def call_openai_api(prompt: str, model_name: str):
    try:
        out = (
            openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
            )
            .choices[0]
            .message.content
        )
        return out
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def nshot(
    model_name,
    model_trio,
    n,
    trie_search,
    show_labels,
    train,
    test,
    labels,
    max_tokens,
    device
):
    if "gpt" not in model_name:
        model, tokenizer, generation_config = model_trio
        model.eval()

    if trie_search:
        if "gpt" in model_name:
            raise NotImplementedError(
                "GPT can not be used with trie search due to limited API capabilities"
            )
        trie = Trie(
            bos_token_id=tokenizer.bos_token_id,
            sep_token_id=tokenizer.sep_token_id,
            eos_token_id=tokenizer.eos_token_id,
            col_token_id=tokenizer.col_token_id,
            pad_token_id=tokenizer.pad_token_id,
            sequences=[tokenizer.encode(lab) for lab in labels],
        )
    preds = []
    if "gpt" in model_name:
        tot = 0
    for row in tqdm(test):
        comp = row[0]
        if train.args.dataset == "industries":
            prompt = "You are an investment professional. Your task is to classify a company, based on its legal name, keywords, and a description, into one or multiple industry sectors."
        elif train.args.dataset == "hatespeech":
            prompt = "You are a social media platform moderator. Your task is to classify a comment into one or multiple categories."
        if show_labels:
            prompt += " The possible labels are: "
            for index, lab in enumerate(labels):
                prompt += f"({index + 1}) {lab}; "
        if n:
            if n == 1:
                prompt += "\nOne example would be:"
            else:
                prompt += "\nSome examples would be:"
            example_indices = random.sample(range(len(train)), n)
            for i in example_indices:
                prompt += f"\n{train[i][0]} {train[i][1]}"

        prompt += "\nThe example to classify is the following: "
        prompt += comp
        if "gpt" in model_name:
            tot += len(prompt)
            success = False
            while not success:
                out = call_openai_api(prompt, model_name)
                if out is not None:
                    success = True
                else:
                    print("Retrying after an error...")
                    time.sleep(60)
            if tot > 500000:
                time.sleep(30)
                tot = 0
        else:
            inputs = tokenizer(
                prompt, return_tensors="pt", max_length=2048, truncation=False
            ).to(device)
            if trie_search:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    generation_config=generation_config,
                    pad_token_id=generation_config.eos_token_id,
                    prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(
                        batch_id, sent.tolist()
                    ),
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    generation_config=generation_config,
                    pad_token_id=generation_config.eos_token_id,
                    do_sample=True,
                )
            out = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        if not model.config.is_encoder_decoder:
            out = out[len(prompt) :]
        preds.append([1 if lab.lower() in out.lower() else 0 for lab in labels])
    
    report = sector_report(test.labels, preds, labels=labels, display=True)
    
    return report, test.labels, preds



def nshot_exp(
    model_name: str,
    ns: List[int],
    trie_search_options: List[bool],
    show_labels_options: List[bool],
    train: MultiLabelT2TDataset,
    test: MultiLabelT2TDataset,
    labels: List[str],
    out_path: str,
    device: str,
    load_in_8bit: bool,
):
    model_trio = (
        create_model(model_name, False, device, load_in_8bit)
        if "gpt" not in model_name
        else None
    )
    results = {"model_name": model_name}
    dir_path = f"{out_path}/{model_name}"
    os.makedirs(dir_path, exist_ok=True)
    file_exists = True
    file_num = 0
    while file_exists:
        filepath = f"{dir_path}/{file_num}.json"
        file_exists = os.path.isfile(filepath)
        file_num += 1
    for n in ns:
        for trie_seach in trie_search_options:
            for show_labels in show_labels_options:
                trie_search_str = "_trie-search" if trie_seach else ""
                show_labels_str = "_show-labels" if show_labels else ""
                print(f"\n<<<|{n}-shot{trie_search_str}{show_labels_str}|>>>\n")
                results_dict, _, _ = nshot(
                    model_name,
                    model_trio=model_trio,
                    n=n,
                    trie_search=trie_seach,
                    show_labels=show_labels,
                    train=train,
                    test=test,
                    labels=labels,
                    device=device,
                )
                results[f"{n}-shot{trie_search_str}{show_labels_str}"] = results_dict
                with open(filepath, "w") as fp:
                    json.dump(results, fp, indent=4)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="hatespeech",
        choices=["hatespeech", "industries"],
    )
    parser.add_argument("--model_name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--trie_search", type=str, default="both")
    parser.add_argument("--show_labels", type=str, default="both")
    parser.add_argument("--labels", type=str, default="keywords")
    parser.add_argument("--load_in_8bit", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = MultiLabelT2TDataset("train_preprocessed.json", args)
    test = MultiLabelT2TDataset("test_preprocessed.json", args)

    ns = [0, 1, 2, 4, 8]
    trie_search_options = [True, False]
    show_labels_options = [True, False]
    if args.n:
        ns = [args.n]
    if args.trie_search == "True":
        trie_search_options = [True]
    elif args.trie_search == "False":
        trie_search_options = [False]
    if args.show_labels == "True":
        show_labels_options = [True]
    elif args.show_labels == "False":
        show_labels_options = [False]

    nshot_exp(
        args.model_name,
        ns=ns,
        trie_search_options=trie_search_options,
        show_labels_options=show_labels_options,
        train=train,
        test=test,
        labels=train.labels.columns,
        out_path=BASE_DIR / "sectors/experiments/nshot/nshot_results",
        device=device,
        load_in_8bit=args.load_in_8bit,
    )
