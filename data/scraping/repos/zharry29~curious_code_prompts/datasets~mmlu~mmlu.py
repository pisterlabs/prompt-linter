import argparse
import openai
import os
import numpy as np
import pandas as pd
import time

from crop import crop

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
    # print("Index " + str(idx))
    # print("DF len " + str(len(df)))
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_text_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        #print("I am in generating text prompt on train df")
        prompt += format_example(train_df, i)
    return prompt

def gen_prompt(args, train_df, subject, k=-1):
    if args.prompt == 'text':
        return gen_text_prompt(train_df, subject, k)
    if args.prompt == 'code':
        print("Code prompt not implemented! - Using text prompt")
        return gen_text_prompt(train_df, subject, k)

def eval(args, subject, engine, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        #print("I am in generating text prompt on test df")
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(args, dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(args, dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]
        #print(k)
        while True:
            try:
                c = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    max_tokens=1,
                    logprobs=100,
                    temperature=0,
                    echo=True
                )
                break
            except:
                print("pausing")
                time.sleep(1)
                continue

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

def main(args):
    if not os.path.exists(args.data_dir):
        print("No data found -- Downloading dataset")
        import wget, tarfile
        os.mkdir(args.data_dir)
        wget.download("https://people.eecs.berkeley.edu/~hendrycks/data.tar", "data.tar")
        data = tarfile.open('data.tar')
        data.extractall(args.data_dir)
        data.close()
        os.remove('data.tar')
    args.data_dir += '/data'

    engines = args.model
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    for engine in engines:
        if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(engine))):
            os.mkdir(os.path.join(args.save_dir, "results_{}".format(engine)))

    print(subjects)
    print(args)

    for engine in engines:
        print(engine)
        all_cors = []

        for subject in subjects:
            dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)
            val_df = pd.read_csv(os.path.join(args.data_dir, "val", subject + "_val.csv"), header=None)
            dev_df = pd.concat([dev_df, val_df])
            test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

            cors, acc, probs = eval(args, subject, engine, dev_df, test_df)
            all_cors.append(cors)

            test_df["{}_correct".format(engine)] = cors
            for j in range(probs.shape[1]):
                choice = choices[j]
                test_df["{}_choice{}_probs".format(engine, choice)] = probs[:, j]
            test_df.to_csv(os.path.join(args.save_dir, "results_{}".format(engine), "{}.csv".format(subject)), index=None)

        weighted_acc = np.mean(np.concatenate(all_cors))
        print("Average accuracy: {:.3f}".format(weighted_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-e", choices=["davinci", "curie", "babbage", "ada"],
                        default=["davinci", "curie", "babbage", "ada"], nargs="+")
    parser.add_argument('--prompt', required=True, type=str, help='Either text or code.')
    parser.add_argument('--key', required=True, type=str, help='The name of the OpenAI API key file.')
    args = parser.parse_args()

    openai.api_key = open(f'../../_private/{args.key}.key').read()

    main(args)