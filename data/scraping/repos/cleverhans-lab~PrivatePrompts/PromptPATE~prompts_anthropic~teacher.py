import anthropic
#import dataset
from datasets import load_dataset
import numpy as np
import argparse
import re
import random
import time

datasets = load_dataset("trec", split="train")
#print(medical_questions)
filename = f"logging/teacher_trec_{random.randint(0, 100000)}.log"
logf = open(filename, "a")
    #save_results(p, output_file=f)
    #f.close()
public_sentence = datasets["text"][:200]
#hypothesis = datasets["hypothesis"][:400]
public_label = datasets["coarse_label"][:200]

with open("data_public/qqp.txt", "r") as f:
    public_sentence_ood = [line.strip() for line in f.readlines()]
#public_sentence_ood = np.loadtxt("data_public/qqp.txt")
public_sentence_ood = public_sentence_ood[:200]

example_sentence = datasets["text"][200:]
example_label = datasets["coarse_label"][200:]

def check_duplicate(arr_a, arr_b):
    for e in arr_a:
        if e in arr_b:
            return True
    return False

#example_label = datasets["label"][400:500]
def extract(response):
    description_regex = re.compile(r'<answer>(.+?)</answer>', re.DOTALL)
    match = description_regex.search(response)

    if match:
        description_content = match.group(1)
        return description_content.strip()
    else:
        return " "

selected_prompts = np.random.choice(len(datasets)-500, 1600, replace = False)
for j in range(400):
    index = selected_prompts[j*4:(j+1)*4]
    predictions = np.zeros(200)
    for i in range(200):
        #index = selected_prompts[j*4:(j+1)*4]
        responsed = False
        while not responsed:
            try:
                response = client.completion(prompt=f"{anthropic.HUMAN_PROMPT} Classify the questions based on whether their answer type is a 0 (Abbreviation), 1 (Entity), 2 (Description), 3 (Human), 4 (Location), or 5 (Number). Several examples are provided to help you with the task. Please put your answer in the <answer> tag. <sentence>{example_sentence[index[0]]}</sentence>\n<answer>{example_label[index[0]]}</answer>\n\n<sentence>{example_sentence[index[1]]}</sentence>\n<answer>{example_label[index[1]]}</answer>\n\n<sentence>{example_sentence[index[2]]}</sentence>\n<answer>{example_label[index[2]]}</answer>\n\n<sentence>{example_sentence[index[3]]}</sentence>\n<answer>{example_label[index[3]]}</answer>\n\n<sentence>{public_sentence[i]}</sentence>\n{anthropic.AI_PROMPT}\n", model="claude-v1", max_tokens_to_sample=10, temperature = 0)
                responsed = True
            except:
                time.sleep(5)
        print(i, extract(response["completion"]))
        try:
            predictions[i] = int(extract(response["completion"]))
        except:
            #print(i, response["completion"])
            predictions[i] = -2
    accuracy = np.mean(predictions == public_label)
    predictions = predictions.astype(int).tolist()
    print("validation accuracy is ", file=logf, flush=True)
    print(accuracy, file=logf, flush=True)
    print("labels for the iid public set", file=logf, flush=True)
    print(predictions, file=logf, flush=True)
    print("labels for the iid public set", flush=True)
    print(predictions, flush=True)
    predictions = [0] * 200
    for i in range(200):
        #index = selected_prompts[j*4:(j+1)*4]
        responsed = False
        while not responsed:
            try:
                response = client.completion(prompt=f"{anthropic.HUMAN_PROMPT} Classify the questions based on whether their answer type is a 0 (Abbreviation), 1 (Entity), 2 (Description), 3 (Human), 4 (Location), or 5 (Number). Several examples are provided to help you with the task. Please put your answer in the <answer> tag. <sentence>{example_sentence[index[0]]}</sentence>\n<answer>{example_label[index[0]]}</answer>\n\n<sentence>{example_sentence[index[1]]}</sentence>\n<answer>{example_label[index[1]]}</answer>\n\n<sentence>{example_sentence[index[2]]}</sentence>\n<answer>{example_label[index[2]]}</answer>\n\n<sentence>{example_sentence[index[3]]}</sentence>\n<answer>{example_label[index[3]]}</answer>\n\n<sentence>{public_sentence_ood[i]}</sentence>\n{anthropic.AI_PROMPT}\n", model="claude-v1", max_tokens_to_sample=10, temperature = 0)
                responsed = True
            except:
                time.sleep(5)

        print(i, extract(response["completion"]))
        try:
            predictions[i] = int(extract(response["completion"]))
        except:
            #print(i, response["completion"])
            predictions[i] = -2
    #accuracy[j] = np.mean(predictions == label)
    print("labels for the ood public set", file=logf, flush=True)
    print(predictions, file=logf, flush=True)
    print("labels for the ood public set", flush=True)
    print(predictions, flush=True)


