import argparse
import json
import logging
import os
import sys
import time
from functools import lru_cache

import numpy as np
import openai
import pandas as pd
import regex as re
import requests
import torch
import tqdm
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

choices = ["A", "B", "C", "D"]


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator
    return softmax


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval_hf(subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    all_explain = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = 3
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        label = test_df.iloc[i, test_df.shape[1] - 1]

        decoder_input_ids = tokenizer("", return_tensors="pt").input_ids.cuda()
        decoder_input_ids = model._shift_right(decoder_input_ids)
        logits = model(
            input_ids=input_ids, decoder_input_ids=decoder_input_ids
        ).logits.flatten()

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[0]],
                        logits[tokenizer("B").input_ids[0]],
                        logits[tokenizer("C").input_ids[0]],
                        logits[tokenizer("D").input_ids[0]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        train_prompt = gen_prompt(dev_df, subject, 2)
        prompt = train_prompt + prompt_end + "Please explain the answer in a sentence."
        input_ids = tokenizer(prompt, return_tensors="pt").to(device)
        generated_tokens = model.generate(
            **input_ids, early_stopping=False, max_length=200
        )
        expl = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        all_explain.append(expl)
        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)
    all_explain = np.array(all_explain)
    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs, all_explain


def eval_model_hf(model_hf, data_dir, save_dir, n_gpu=1):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_hf)
    tokenizer = AutoTokenizer.from_pretrained(model_hf)
    heads_per_gpu = len(model.encoder.block) // n_gpu
    device_map = {
        gpu: list(
            range(
                0 + (gpu * heads_per_gpu),
                (0 + (gpu * heads_per_gpu)) + heads_per_gpu,
            )
        )
        for gpu in range(n_gpu)
    }
    model.parallelize(device_map)
    model.eval()
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, "results_{}".format(model_hf))):
        os.makedirs(os.path.join(save_dir, "results_{}".format(model_hf)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        print(f" subject {subject}")
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
        )[:5]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs, all_explain = eval_hf(
            subject, model, tokenizer, dev_df, test_df
        )
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(model_hf)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(model_hf, choice)] = probs[:, j]
        test_df["explanations"] = all_explain
        test_df.to_csv(
            os.path.join(
                save_dir, "results_{}".format(model_hf), "{}.csv".format(subject)
            ),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))


def _get_encoder(subdir):
    print("Downloading encoder and vocab to ", subdir)
    for filename in ["encoder.json", "vocab.bpe"]:
        r = requests.get(
            "https://openaipublic.blob.core.windows.net/gpt-2/"
            + subdir
            + "/"
            + filename,
            stream=True,
        )
        with open(os.path.join(subdir, filename), "wb") as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(
                ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True
            ) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges, errors="replace"):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            "utf-8", errors=self.errors
        )
        return text


def get_encoder(model_name):
    subdir = os.path.join("models", model_name)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    if not os.path.exists(os.path.join(subdir, "encoder.json")):
        _get_encoder(subdir)

    subdir = subdir.replace("\\", "/")  # needed for Windows

    with open(os.path.join(subdir, "encoder.json"), "r") as f:
        encoder = json.load(f)
    with open(os.path.join(subdir, "vocab.bpe"), "r", encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )


enc = get_encoder("124M")


def crop_prompt(prompt: str):
    global enc

    cropped_prompt = enc.decode(enc.encode(prompt)[:2048])
    return cropped_prompt


def crop(s):
    prompt = crop_prompt(s)
    return prompt


def get_completion(prompt):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                    # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    # {"role": "user", "content": "Where was it played?"}
                ],
            )
            logging.info("Called OPENAI API")
            return response["choices"][0]["message"]["content"]
        except:
            print("pausing")
            time.sleep(0.3)
            continue


def eval_openai(subject, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]
    all_explain = []
    preds = []
    for i in tqdm(range(test_df.shape[0])):
        # get prompt and make sure it fits
        k = 1
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1] - 1]

        probs = softmax(np.array([0, 0, 0, 0]))
        pred = get_completion(prompt)
        preds.append(pred)
        prompt = (
            train_prompt
            + prompt_end
            + "Please explain the answer in a sentence to enable someone to verify your answer:"
        )
        expl = get_completion(prompt)
        all_explain.append(expl)

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)
    all_explain = np.array(all_explain)
    all_probs = np.array(all_probs)
    preds = np.array(preds)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs, all_explain, preds


def eval_model_openai(model_hf, data_dir, save_dir):
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, "results_{}".format(model_hf))):
        os.makedirs(os.path.join(save_dir, "results_{}".format(model_hf)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}
    for subject in tqdm(subjects):
        print(f" subject {subject}")
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
        )[:5]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )

        test_df = test_df.sample(frac=1).reset_index(drop=True)
        test_df = test_df[:150]  # [:150]
        print(subject)
        print(len(test_df))

        cors, acc, probs, all_explain, preds = eval_openai(subject, dev_df, test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(model_hf)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(model_hf, choice)] = probs[:, j]

        test_df["explanations"] = all_explain
        test_df["predss"] = preds
        test_df.to_csv(
            os.path.join(
                save_dir, "results_{}".format(model_hf), "{}.csv".format(subject)
            ),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))


# below not used:


def eval_obqa(subject, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]
    all_explain = []
    preds = []
    k = 0
    for i in tqdm(range(test_df.shape[0])):
        # get prompt and make sure it fits
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = prompt_end

        label = test_df.iloc[i, test_df.shape[1] - 1]

        probs = softmax(np.array([0, 0, 0, 0]))
        pred = get_completion(prompt)
        preds.append(pred)
        print(pred)
        print(label)
        prompt = train_prompt + prompt_end + "Please explain the answer in a sentence."
        expl = get_completion(prompt)
        all_explain.append(expl)

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)
    all_explain = np.array(all_explain)
    all_probs = np.array(all_probs)
    preds = np.array(preds)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs, all_explain, preds


def eval_model_obqa():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, "results_{}".format(model_hf))):
        os.makedirs(os.path.join(save_dir, "results_{}".format(model_hf)))

    all_cors = []

    test_df = pd.read_csv(data_dir, header=None)
    subject = "obqa"
    dev_df = None
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df[:150]
    print(len(test_df))

    cors, acc, probs, all_explain, preds = eval_obqa(subject, dev_df, test_df)

    test_df["{}_correct".format(model_hf)] = cors
    for j in range(probs.shape[1]):
        choice = choices[j]
        test_df["{}_choice{}_probs".format(model_hf, choice)] = probs[:, j]

    test_df["explanations"] = all_explain
    test_df["predss"] = preds
    test_df.to_csv(
        os.path.join(save_dir, "results_{}".format(model_hf), "{}.csv".format(subject)),
        index=None,
    )


subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": [
        "physics",
        "chemistry",
        "biology",
        "computer science",
        "math",
        "engineering",
    ],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}
