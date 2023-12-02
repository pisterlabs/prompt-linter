import anthropic
#import dataset
from datasets import load_dataset
import numpy as np
import argparse
import re
import random
import time

parser = argparse.ArgumentParser()
parser.add_argument('--public_index', dest='public_index', action='store', required=True)
parser.add_argument('--public_label', dest='public_label', action='store', required=True)

args = parser.parse_args()
args = vars(args)

test_datasets = load_dataset("trec", split="test")
#print(medical_questions)
public_datasets = "trec"
if "qqp" in args["public_index"]:
    public_datasets = "qqp"
filename = f"logging/student_trec_{public_datasets}_{random.randint(0, 100000)}.log"
logf = open(filename, "a")
    #save_results(p, output_file=f)
    #f.close()
test_sentence = test_datasets["text"][:300]
test_labels = test_datasets["coarse_label"][:300]

valid_sentence = None
valid_labels = np.loadtxt(args["public_label"], dtype=int)
valid_index = np.loadtxt(args["public_index"], dtype=int)

if "qqp" in args["public_index"]:
    with open("data_public/qqp.txt", "r") as f:
        valid_sentence = [line.strip() for line in f.readlines()]
else:
    datasets = load_dataset("trec", split="train")
    valid_sentence = datasets["text"]
valid_sentence = valid_sentence[:200]
valid_sentence = [valid_sentence[i] for i in valid_index]

#example_label = datasets["label"][400:500]
def extract(response):
    description_regex = re.compile(r'<answer>(.+?)</answer>', re.DOTALL)
    match = description_regex.search(response)

    if match:
        description_content = match.group(1)
        return description_content.strip()
    else:
        return " "
best_prompt_index = 0
best_validation = 0
for j in range(50):
    index = np.random.choice(len(valid_sentence), 4, replace=False)
    predictions = np.zeros(len(valid_sentence))
    for i in range(len(valid_sentence)):
        #index = selected_prompts[j*4:(j+1)*4]
        responsed = False
        while not responsed:
            try:
                response = client.completion(prompt=f"{anthropic.HUMAN_PROMPT} Classify the questions based on whether their answer type is a 0 (Abbreviation), 1 (Entity), 2 (Description), 3 (Human), 4 (Location), or 5 (Number). Several examples are provided to help you with the task. Please put your answer in the <answer> tag. <sentence>{valid_sentence[index[0]]}</sentence>\n<answer>{valid_labels[index[0]]}</answer>\n\n<sentence>{valid_sentence[index[1]]}</sentence>\n<answer>{valid_labels[index[1]]}</answer>\n\n<sentence>{valid_sentence[index[2]]}</sentence>\n<answer>{valid_labels[index[2]]}</answer>\n\n<sentence>{valid_sentence[index[3]]}</sentence>\n<answer>{valid_labels[index[3]]}</answer>\n\n<sentence>{valid_sentence[i]}</sentence>\n{anthropic.AI_PROMPT}<answer>", model="claude-v1", max_tokens_to_sample=1, temperature = 0)
                responsed = True
            except:
                time.sleep(5)
        print(i, response["completion"])
        try:
            predictions[i] = int(response["completion"])
        except:
            #print(i, response["completion"])
            predictions[i] = -2
    accuracy = np.mean(predictions == valid_labels)
    if accuracy > best_validation:
        best_validation = accuracy
        best_prompt_index = index
    print("validation accuracy is " + str(accuracy), file=logf, flush=True)
print("best validation accuracy is " + str(best_validation), file=logf, flush=True)
predictions = np.zeros(len(test_sentence))
for i in range(len(test_sentence)):
    # index = selected_prompts[j*4:(j+1)*4]
    responsed = False
    while not responsed:
        try:
            response = client.completion(
                prompt=f"{anthropic.HUMAN_PROMPT} Classify the questions based on whether their answer type is a 0 (Abbreviation), 1 (Entity), 2 (Description), 3 (Human), 4 (Location), or 5 (Number). Several examples are provided to help you with the task. Please put your answer in the <answer> tag. <sentence>{valid_sentence[index[0]]}</sentence>\n<answer>{valid_labels[index[0]]}</answer>\n\n<sentence>{valid_sentence[index[1]]}</sentence>\n<answer>{valid_labels[index[1]]}</answer>\n\n<sentence>{valid_sentence[index[2]]}</sentence>\n<answer>{valid_labels[index[2]]}</answer>\n\n<sentence>{valid_sentence[index[3]]}</sentence>\n<answer>{valid_labels[index[3]]}</answer>\n\n<sentence>{test_sentence[i]}</sentence>\n{anthropic.AI_PROMPT}<answer>",
                model="claude-v1", max_tokens_to_sample=1, temperature=0)
            responsed = True
        except:
            time.sleep(5)
    print(i, response["completion"])
    try:
        predictions[i] = int(response["completion"])
    except:
        # print(i, response["completion"])
        predictions[i] = -2

accuracy = np.mean(predictions == test_labels)
print("test accuracy is " + str(accuracy), file=logf, flush=True)


