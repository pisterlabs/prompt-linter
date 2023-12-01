import argparse
import openai
import os
import numpy as np
import pandas as pd
import time

from utils.crop import crop

openai.api_key = "openai_api_key"
choices = ["A", "B", "C", "D"]


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
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
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def eval(args, subject, engine, dev_df, test_df):

    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]

        c = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=1,
            logprobs=100,
            temperature=0,
            echo=True
        )
         
        lprobs = []
        for ans in answers:
            try:
                lprobs.append(c["choices"][0]["logprobs"]["top_logprobs"][-1][" {}".format(ans)])
            except:
                print("Warning: {} not found. Artificially adding log prob of -100.".format(ans))
                lprobs.append(-100)
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

def test_dataset(data_args, subject, engine, dev_df, test_df):
    cors, acc, probs = eval(data_args, subject, engine, dev_df, test_df)

    # save results
    test_df["{}_correct".format(engine)] = cors
    for j in range(probs.shape[1]):
        choice = choices[j]
        test_df["{}_choice{}_probs".format(engine, choice)] = probs[:, j]
    test_df.to_csv(os.path.join(data_args.save_dir, "results_{}".format(engine), "{}.csv".format(subject)), index=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")

    data_args = parser.parse_args()
    engine = "davinci"
    # file_names = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])
    subject = "machine_learning"
  
    # read in data
    dev_df = pd.read_csv(os.path.join(data_args.data_dir, "dev", subject + "_dev.csv"), header=None)[:data_args.ntrain]
    test_df = pd.read_csv(os.path.join(data_args.data_dir, "test", subject + "_test.csv"), header=None)[:1]

    # get prompt and make sure it fits
    k = data_args.ntrain
    prompt_end = format_example(test_df, 0, include_answer=False)
    train_prompt = gen_prompt(dev_df, subject, k)
    prompt = train_prompt + prompt_end

    while crop(prompt) != prompt:
        k -= 1
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

    label = test_df.iloc[0, test_df.shape[1]-1]

    c = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=1,
        logprobs=100,
        temperature=0,
        echo=True
    )
        
    lprobs = []

    answers = choices[:test_df.shape[1]-2]
    for ans in answers:
        try:
            lprobs.append(c["choices"][0]["logprobs"]["top_logprobs"][-1][" {}".format(ans)])
        except:
            print("Warning: {} not found. Artificially adding log prob of -100.".format(ans))
            lprobs.append(-100)
    pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
    probs = softmax(np.array(lprobs))

    cor = pred == label


